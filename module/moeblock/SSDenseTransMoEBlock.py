import torch
import torch.nn as nn
import torch.nn.functional as F

from module.linear.SSTransLinear import SSTransLinear
from module.linear.SPDenseTransLinear import SPDenseTransLinear

from module.moeblock.MoEBlock import MoEBlock, MLP

class SSMLP(nn.Module):
    def __init__(
        self, 
        ffn_dim: int,
        hidden_dim: int,
        act_fn: nn.Module = nn.SiLU(),
        original: MLP = None
    ):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.hidden_dim =  hidden_dim

        # w1 w3是SS的矩阵乘，input带有expert选择，hidden_states是完整的中间结果，返回结果为压缩的稠密矩阵
        # w2是SD的矩阵乘
        # w1 w2 w3 的权重都会根据要求进行稀疏化裁剪
        self.w1 = SSTransLinear(original, in_features=self.ffn_dim, out_features=self.hidden_dim)
        self.w2 = SPDenseTransLinear(original, in_features=self.hidden_dim, out_features=self.ffn_dim)
        self.w3 = SSTransLinear(original, in_features=self.ffn_dim, out_features=self.hidden_dim)

        self.act_fn = act_fn

    def forward(self, hidden_states, input_idx, routing_weights):
        batch_size = input_idx.shape[0]
        current_hidden_states = self.act_fn(self.w1(hidden_states, input_idx)) * self.w3(hidden_states, input_idx)
        current_hidden_states = self.w2(current_hidden_states)
        return routing_weights * current_hidden_states[0:batch_size, :]


class SSDenseTransMoEBlock(nn.Module):
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

    def __init__(
        self, 
        num_experts: int,
        top_k: int,
        ffn_dim: int,
        hidden_dim: int,
        original: MoEBlock = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # gating
        if original is not None:
            self.gate = original.gate
        else:
            self.gate = nn.Linear(self.ffn_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([SSMLP(self.ffn_dim, self.hidden_dim, original=original) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # print(hidden_states.shape, hidden_states.dtype)
        origin_shape = hidden_states.shape
        # _, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.ffn_dim)
        batch_size, hidden_dim = hidden_states.shape
        # router_logits: (batch * sequence_length, n_experts)
        # router_logits, _ = self.gate(hidden_states)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        final_hidden_states = torch.zeros(
            (batch_size, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            # for expert_idx in range(1):
            # print("In expert: ", expert_idx, " / ", self.num_experts)
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
            # current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            # current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])

            # 实际计算时hidden_states的选择在算子中进行
            current_hidden_states = expert_layer(hidden_states, top_x, routing_weights[top_x_list, idx_list, None])
            # expert_layer(hidden_states, top_x, routing_weights[top_x_list, idx_list, None])

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            # final_hidden_states = final_hidden_states + current_hidden_states.to(hidden_states.dtype)
        final_hidden_states = final_hidden_states.reshape(origin_shape)
        return final_hidden_states
