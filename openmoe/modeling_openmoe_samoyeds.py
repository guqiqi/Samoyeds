#
# Copyright (c) 2025 Chenpeng Wu (cpwu_sjtu@sjtu.edu.cn), Qiqi Gu (qiqi.gu@sjtu.edu.cn). 
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Optional, Tuple

from colossalai.moe.layers import SparseMLP
from colossalai.moe.experts import MLPExperts
from colossalai.moe._operation import MoeInGradScaler, MoeOutGradScaler
from colossalai.moe.routers import MoeRouter, get_router_cls
from colossalai.moe.utils import get_noise_generator

from module.linear.SPDenseWeightedTransLinear import SPDenseWeightedLinear
from module.linear.SSTransLinear import SSTransLinear

from openmoe.router import Top2Router

class Expert(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        act,
        drop
    ):
        super().__init__()
        
        self.act = act
        self.drop = drop
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.gate = SSTransLinear(nn.Linear(self.hidden_size, self.intermediate_size*2, bias=False))
        self.up = SSTransLinear(nn.Linear(self.hidden_size, self.intermediate_size, bias=False))
        self.down = SPDenseWeightedLinear(nn.Linear(self.intermediate_size, self.hidden_size, bias=False))
        
    def forward(
        self,
        x: torch.Tensor,
        top_x: torch.Tensor, 
        weight: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = top_x.size(0)
        
        x_gate = self.gate(x, top_x)
        x_up = self.up(x, top_x)
        x_intermediate = self.act(x_gate) * x_up
        x_intermediate = self.drop(x_intermediate)
        
        current_hidden_states = self.down(x_intermediate, weight)
        
        return current_hidden_states[0:batch_size, :]    

class SSMLPExperts(nn.Module):
    def __init__(
        self,
        original: MLPExperts
    ):
        super().__init__()
        self.expert_parallel = original.expert_parallel
        self.num_total_experts = original.num_total_experts
        self.gated = original.gated
        self.use_kernel = original.use_kernel
        self.hidden_size = original.hidden_size
        self.intermediate_size = original.intermediate_size

        self.num_local_experts = self.num_total_experts
        self.ep_size = 1
        
        self.experts = nn.ModuleList(Expert(self.hidden_size, self.intermediate_size, original.act, original.drop) for i in range(self.num_total_experts))
    
    def forward(
        self,
        x: torch.Tensor,
        used_capacity: list,
        top_x_list: list, 
        weight_list: list,
        param_slice: Tuple[slice] = (slice(None),),
        use_sparse: bool = True,
    ) -> torch.Tensor:
        """
        forward: hidden_size --> intermediate_size --> hidden_size

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size * sequence_len, hidden_size)

        Returns:
            torch.Tensor: The output tensor of shape (batch_size * sequence_len, hidden_size)
        """
        # x = MoeInGradScaler.apply(x, self.ep_size)
        e = len(top_x_list)
        h = x.size(-1)
        batch_size = x.size(0)
        
        final_hidden_states = torch.zeros(
            (batch_size, h), dtype=x.dtype, device=x.device
        )

        for i in range(e):
            if used_capacity[i] == 0:
                continue
            
            current_hidden_states = self.experts[i](x, top_x_list[i], weight_list[i])
            final_hidden_states.index_add_(0, top_x_list[i], current_hidden_states.to(x.dtype))
            
            
        # x_gate = [torch.mm(x[i], self.wi_gate[param_slice][i]) for i in range(e)]
        # x_gate = [sparse_dense_mul_transpose_dense_dispatch(self.wi_gate_weight[i], self.wi_gate_indices[i], self.wi_gate_metadata[i], x[i], self.intermediate_size * 2, self.hidden_size, x[i].shape[0]) for i in range(e)]
        
        # # x_up = [torch.mm(x[i], self.wi_up[param_slice][i]) for i in range(e)]
        # x_up = [sparse_dense_mul_transpose_dense_dispatch(self.wi_up_weight[i], self.wi_up_indices[i], self.wi_up_metadata[i], x[i], self.intermediate_size, self.hidden_size, x[i].shape[0]) for i in range(e)]
        
        # x = [self.act(x_gate[i]) * x_up[i] for i in range(e)]
        
        # x = [self.drop(x[i]) for i in range(e)]
        # # x = [torch.mm(x[i], self.wo[param_slice][i]) for i in range(e)]
        # x = [sparse_dense_mul_transpose_dense_dispatch(self.wo_weight[i], self.wo_indices[i], self.wo_metadata[i], x[i], self.hidden_size, self.intermediate_size, x[i].shape[0]) for i in range(e)]

        # x = torch.cat([x[i].unsqueeze(0) for i in range(e)], dim=0)
        # x = x.reshape(inshape)
        # x = x.transpose(0, 1).contiguous()
        # x = MoeOutGradScaler.apply(x, self.ep_size)
        return final_hidden_states

class SSSparseMLP(nn.Module):
    def __init__(self, original: SparseMLP):
        super().__init__()
        # self.ffn_dim = original.ffn_dim
        # self.hidden_dim = original.hidden_dim
        
        self.hidden_size = original.hidden_size
        self.intermediate_size = original.intermediate_size
        self.num_experts = original.num_experts
        self.gated = original.gated
        self.enable_kernel = original.enable_kernel
        self.enable_comm_overlap = original.enable_comm_overlap
        self.expert_parallel = original.expert_parallel
        
        noisy_func = None
        # router_cls = get_router_cls(original.topk)
        # self.topk = original.topk
        # self.router = original.router
        router_cls = Top2Router
        self.topk = original.topk
        self.router: MoeRouter = router_cls(
            capacity_factor_train=1.25,
            capacity_factor_eval=2.0,
            min_capacity=4,
            noisy_func=noisy_func,
            drop_tks=True,
        )
        
        self.gate_weight = original.gate_weight
        
        self.experts = SSMLPExperts(original.experts)

        self.num_local_experts = self.experts.num_local_experts

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): The input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # print(inputs.shape)
        # reshape the input tokens
        tokens = inputs.reshape(-1, self.hidden_size)

        # the data type of the inputs in the gating should be fp32
        fp32_input = tokens.to(torch.float)
        fp32_weight = self.gate_weight.to(torch.float)
        gate_output = F.linear(fp32_input, fp32_weight)

        # the result from the router
        used_capacity, weight_list, top_x_list = self.router(
            inputs=gate_output, use_kernel=self.enable_kernel, ep_group=None)

        expert_output = self._local_process(tokens, used_capacity, top_x_list, weight_list)

        ans = expert_output.reshape(inputs.shape)
        return ans
    
    def _local_process(self, expert_in: torch.Tensor, used_capacity: list, top_x_list: list, weight_list: list) -> torch.Tensor:
        # expert_in = expert_in.unsqueeze(0)
        expert_out = self.experts(expert_in, used_capacity, top_x_list, weight_list)
        return expert_out
    
def sparsemoeblock_to_ss(mod):
    if isinstance(mod, SparseMLP):
        print("in module replace")
        return SSSparseMLP(mod)

    for name, m in mod.named_children():
        if isinstance(m, SSSparseMLP):
            continue
        # if isinstance(m, torch.nn.Linear):
        if isinstance(m, SparseMLP):
            setattr(mod, name, SSSparseMLP(m))
        elif m is not mod:
            sparsemoeblock_to_ss(m)

    return mod