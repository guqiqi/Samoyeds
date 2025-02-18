from typing import List, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from vllm.model_executor.models.deepseek import DeepseekMLP

try:
    import megablocks.ops as ops
except ImportError:
    print(
        "MegaBlocks not found. Please install it by `pip install megablocks`. "
        "Note that MegaBlocks depends on mosaicml-turbo, which only supports "
        "Python 3.10 for now.")
try:
    import stk
except ImportError:
    print(
        "STK not found: please see https://github.com/stanford-futuredata/stk")

from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.utils import set_weight_attrs

KVCache = Tuple[torch.Tensor, torch.Tensor]


def promote_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.view(1) if len(x.size()) == 0 else x

class DeepSeekBlockSparseMoE(nn.Module):
    """
    Built on the paper and library Megablocks as described in
    https://arxiv.org/abs/2211.15841. This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int,
                 top_k: int, n_shared_experts=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_shared_experts = n_shared_experts

        # gating
        self.gate = nn.Linear(self.hidden_dim,
                              self.num_experts,
                              bias=False,
                              device=torch.cuda.current_device())
        
        if self.n_shared_experts is not None:
            intermediate_size = (self.ffn_dim *
                                 self.n_shared_experts)
            self.shared_experts = DeepseekMLP(
                hidden_size=self.hidden_dim,
                intermediate_size=intermediate_size,
                hidden_act="silu",
                reduce_results=False,
            )

        tp_size = get_tensor_model_parallel_world_size()
        assert self.ffn_dim % tp_size == 0
        self.ffn_dim_per_partition = self.ffn_dim // tp_size
        # merged expert weights, all of size  (ffn_dim * n_experts, model_dim)
        self.w1 = nn.Parameter(
            torch.empty(self.ffn_dim_per_partition * self.num_experts,
                        self.hidden_dim,
                        device=torch.cuda.current_device()))
        set_weight_attrs(self.w1, {"weight_loader": self.moe_weight_loader})
        self.w2 = nn.Parameter(
            torch.empty(self.ffn_dim_per_partition * self.num_experts,
                        self.hidden_dim,
                        device=torch.cuda.current_device()))
        set_weight_attrs(self.w2, {"weight_loader": self.moe_weight_loader})
        self.w3 = nn.Parameter(
            torch.empty(self.ffn_dim_per_partition * self.num_experts,
                        self.hidden_dim,
                        device=torch.cuda.current_device()))
        set_weight_attrs(self.w3, {"weight_loader": self.moe_weight_loader})

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort.
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)
        self.blocking = 128
        self.quantize_scatter_num_bits = -1

        # Calculate the number of bits needed to represent the column indices
        # in the intermediate sparse matrix.
        max_column_index = (self.ffn_dim * self.num_experts) // self.blocking
        self.transpose_sort_end_bit = max(
            int(np.ceil(np.log2(max_column_index))), 1)

    def moe_weight_loader(self, param: nn.Parameter,
                          loaded_weight: torch.Tensor) -> None:
        """
        Load the weights for the MoE linear layer.
        """
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = self.ffn_dim_per_partition
        loaded_weight = loaded_weight.view(self.num_experts, self.ffn_dim, -1)
        loaded_weight = loaded_weight[:, shard_size * tp_rank:shard_size *
                                      (tp_rank + 1)]
        loaded_weight = loaded_weight.reshape_as(param)
        param.data.copy_(loaded_weight)

    def sparse_transpose(
            self, size: int, row_indices,
            column_indices) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_columns = size[1] // self.blocking

        # Sort row indices by column indices to get the transposed matrix's
        # column indices.
        #
        # NOTE: Our sort operation uses the same width indices as the input
        # values. To avoid overflow when we have large activation matrices
        # we cast to 32-bit before sorting.
        _, gather_indices = ops.sort(column_indices.int(),
                                     self.transpose_sort_end_bit)

        # There are a constant number of blocks in every row of the sparse
        # matrix. A blocks offset is:
        #
        # row_index * blocks_per_row + column_index % blocks_per_row
        #
        # Once we have the block offsets ordered for transposition we can
        # divide by blocks_per_row to get the transposed column indices.
        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()

        zero = torch.zeros((1, ), dtype=torch.int32, device=row_indices.device)
        nnz_per_column = ops.histogram(column_indices, block_columns)
        nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
        offsets_t = torch.cat([zero, nnz_per_column])
        return column_indices_t, offsets_t, block_offsets_t

    def topology(self, x: torch.Tensor,
                 padded_bins: torch.Tensor) -> "stk.Matrix":
        padded_tokens, _ = x.size()
        assert padded_tokens % self.blocking == 0
        assert self.ffn_dim_per_partition % self.blocking == 0

        # Offsets for the sparse matrix. All rows have the
        # same number of nonzero blocks dictated by the
        # dimensionality of a single expert.
        block_rows = padded_tokens // self.blocking
        blocks_per_row = self.ffn_dim_per_partition // self.blocking
        offsets = torch.arange(
            0,
            block_rows * blocks_per_row + 1,
            blocks_per_row,
            dtype=torch.int32,
            device=x.device,
        )

        # Indices for the sparse matrix. The indices for
        # the intermediate matrix are dynamic depending
        # on the mapping of tokens to experts.
        column_indices = ops.topology(padded_bins, self.blocking, block_rows,
                                      blocks_per_row)

        # TODO(tgale): This is unused. Remove the need for this in stk.
        # For now, use meta init to save the device memory.
        data = torch.empty(
            column_indices.numel(),
            self.blocking,
            self.blocking,
            dtype=x.dtype,
            device="meta",
        )
        shape = (padded_tokens, self.ffn_dim_per_partition * self.num_experts)
        row_indices = stk.ops.row_indices(shape, data, offsets, column_indices)
        column_indices_t, offsets_t, block_offsets_t = self.sparse_transpose(
            shape, row_indices, column_indices)
        return stk.Matrix(
            shape,
            data,
            row_indices,
            column_indices,
            offsets,
            column_indices_t,
            offsets_t,
            block_offsets_t,
        )

    def indices_and_padded_bins(
        self, selected_experts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        selected_experts = selected_experts.int()
        bin_ids, indices = ops.sort(selected_experts, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        tokens_per_expert = ops.histogram(selected_experts, self.num_experts)

        # Round the token counts up to the block size used in
        # the matrix muliplications. Caculate the starting
        # position of each bin.
        padded_tokens_per_expert = ops.round_up(tokens_per_expert,
                                                self.blocking)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = promote_scalar(padded_bins)

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = promote_scalar(bins)
        return indices, bin_ids, bins, padded_bins, tokens_per_expert

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (sequence_length, model_dim)
        gate_logits: (sequence_length, n_experts)
        """
        # optional reshape
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])

        # gate_logits: (sequence_length, n_experts)
        gate_logits = self.gate(x)
        # all_probs: (sequence_length, n_experts) and upcast for softmax
        all_probs = F.softmax(gate_logits, dim=1, dtype=torch.float)
        # weights, selected_experts: (sequence_length, top-k)
        weights, selected_experts = torch.topk(all_probs, self.top_k, dim=-1)
        weights /= weights.sum(dim=-1, keepdim=True)
        weights = weights.flatten().to(x.dtype)
        selected_experts = selected_experts.flatten()

        (indices, bin_ids, bins, padded_bins,
         _) = self.indices_and_padded_bins(selected_experts)

        # Permute tokens and pad to prepare expert computation
        # (top_k * sequence_length + padding, model_dim)
        x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins,
                              self.top_k)

        # Create the sparse matrix topology
        with torch.no_grad():
            topo = self.topology(x, padded_bins)

        # Perform the expert computation
        # First Dense x Dense -> Sparse for w1 and w3,
        # (top_k * sequence_length + padding, ffn_dim * n_experts)
        x = stk.Matrix(
            topo.size(),
            F.silu(stk.ops.sdd(x, self.w1.t(), topo).data) *
            stk.ops.sdd(x, self.w3.t(), topo).data,
            topo.row_indices,
            topo.column_indices,
            topo.offsets,
            topo.column_indices_t,
            topo.offsets_t,
            topo.block_offsets_t,
        )

        # Then Sparse x Dense -> Dense for w2
        # (top_k * sequence_length + padding, model_dim)
        x = stk.ops.dsd(x, self.w2)

        x = tensor_model_parallel_all_reduce(x)

        # Permute back and remove padding
        # (top_k * sequence_length, model_dim)
        x = ops.padded_scatter(
            x,
            indices,
            bin_ids,
            weights,
            bins,
            padded_bins,
            self.top_k,
            # self.quantize_scatter_num_bits,
        )
        
        if self.n_shared_experts is not None:
            x += self.shared_experts(x)
        
        return x.view(*input_shape)