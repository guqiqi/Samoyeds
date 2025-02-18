import torch
import torch.nn as nn
import torch.nn.functional as F

from mixtral.modeling_mixtral import MixtralSparseMoeBlock, MixtralBLockSparseTop2MLP

import spatha
from venom_helper.grouped_nmv_tensor import SrNMTensor, nm_vector_mask_sparsify

import sten

m          = 8
n          = 2
v          = 64

def padding(tensor, align):
    m, n = tensor.shape
    pad_rows = (align - m % align) % align
    padding = torch.zeros((pad_rows, n)).to(torch.float16).cuda()
    padded_tensor = torch.cat((tensor, padding), dim=0)
    return padded_tensor

class NMVectorSparsifier:
    def __init__(self, n, m, tileM):
        self.n = n
        self.m = m
        self.tileM = tileM

    def __call__(self, tensor, grad_fmt=None):
        # uncomment to use magnitude-pruning -> mask, columns
        # mask, columns = nm_vector_mask_sparsify(tensor, sparsifier.n, sparsifier.m, sparsifier.tileM)

        # uncomment to use random pruning (cuSparseLt-like approach) -> mask, columns
        nrows, ncols = tensor.shape
        columns = torch.zeros(nrows // self.tileM, ncols // self.m * 4, dtype=torch.int32)
        columns = columns.reshape((-1, 4)) + torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        columns = columns.reshape((nrows // self.tileM, ncols // self.m * 4))

        mask = torch.zeros(tensor.shape, dtype=tensor.dtype)
        m = torch.cat((torch.tensor([1, 0, 1, 0]), torch.zeros(self.m - 4)), 0)
        mask = mask.reshape(-1, self.tileM, self.m) + m
        mask = mask.reshape(tensor.shape)

        sparse_mtx = sten.SparseTensorWrapper.wrapped_from_dense(
            SrNMTensor(self.n, self.m, self.tileM, tensor, mask, columns, tensor.device),
            tensor,
            grad_fmt,
        )

        return sparse_mtx

def sparse_dense_mul_dispatch(sparse_values, sparse_indices, sparse_metadata, dense, nrows_sp, ncols_sp, ncols_d, m, n, v, nnz, bias):

    dense_ = dense.contiguous()

    output = spatha.spmm(sparse_metadata,  # metadata
                         sparse_indices,   # indices
                         sparse_values,    # values
                         dense_,           # rhs_matrix
                         bias,
                         nrows_sp,         # A_num_rows
                         ncols_sp,         # A_num_cols
                         ncols_d,          # B_num_cols
                         v,                # vec_length
                         n,                # n
                         m,                # m
                         nnz,              # nnz
                         0,                # seed
                         32,               # mbrow
                         4                 # brow
                         )

    return output


class SrnmSpmm(torch.nn.Module):
    def __init__(self, original: torch.nn.Linear):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(original.in_features)).half().cuda()

        # Convert weights from original module to SrNM
        w = NMVectorSparsifier(n, m, v)(original.weight).wrapped_tensor

        self.values = torch.nn.Parameter(w.values)

        self.columns = nn.Parameter(w.columns, requires_grad=False)
        self.metadata = nn.Parameter(w.metadata, requires_grad=False)

        self.nrows_sp = w.nrows
        self.ncols_sp = w.ncols
        self.nnz      = w.nnz

    def forward(self, input):
        flattened_input = torch.flatten(input, start_dim=0, end_dim=-2)

        ncols_d  = flattened_input.shape[0]
        DM, _    = flattened_input.shape

        output = sparse_dense_mul_dispatch(self.values, self.columns, self.metadata, flattened_input, self.nrows_sp, self.ncols_sp,
                                           ncols_d, m, n, v, self.nnz, self.bias).T

        output = output.reshape((*input.shape[0:-1], -1))

        return output

class SPMixtralBLockSparseTop2MLP(nn.Module):
    def __init__(self, original: MixtralBLockSparseTop2MLP):
        super().__init__()
        self.ffn_dim = original.ffn_dim
        self.hidden_dim = original.hidden_dim

        # w1 w3是SS的矩阵乘，input带有expert选择，hidden_states是完整的中间结果，返回结果为压缩的稠密矩阵
        # w2是SD的矩阵乘
        # w1 w2 w3 的权重都会根据要求进行稀疏化裁剪
        self.w1 = SrnmSpmm(original.w1)
        self.w2 = SrnmSpmm(original.w2)
        self.w3 = SrnmSpmm(original.w3)

        self.act_fn = original.act_fn

    def forward(self, hidden_states, routing_weights):
        batch_size = hidden_states.shape[0]
        hidden_states = padding(hidden_states, 64)
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return routing_weights * current_hidden_states[0:batch_size, :]


class SPMixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, original: MixtralSparseMoeBlock):
        super().__init__()
        self.hidden_dim = original.hidden_dim
        self.ffn_dim = original.ffn_dim
        self.num_experts = original.num_experts
        self.top_k = original.top_k

        # gating
        self.gate = original.gate

        self.experts = nn.ModuleList([SPMixtralBLockSparseTop2MLP(expert) for expert in original.experts])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            # for expert_idx in range(1):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])

            # 实际计算时hidden_states的选择在算子中进行
            # current_hidden_states = expert_layer(hidden_states, top_x, routing_weights[top_x_list, idx_list, None])
            # expert_layer(hidden_states, top_x, routing_weights[top_x_list, idx_list, None])

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            # final_hidden_states = final_hidden_states + current_hidden_states.to(hidden_states.dtype)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


def sparsemoeblock_to_sp(mod):
    if isinstance(mod, MixtralSparseMoeBlock):
        print("in module replace")
        return SPMixtralSparseMoeBlock(mod)

    for name, m in mod.named_children():
        if isinstance(m, SPMixtralSparseMoeBlock):
            continue
        # if isinstance(m, torch.nn.Linear):
        if isinstance(m, MixtralSparseMoeBlock):
            setattr(mod, name, SPMixtralSparseMoeBlock(m))
        elif m is not mod:
            sparsemoeblock_to_sp(m)

    return mod
