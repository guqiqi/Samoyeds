from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.moe.routers import MoeRouter
from colossalai.moe._operation import moe_cumsum

class Top2Router(MoeRouter):
    """Top2 router that returns the dispatch mask (batch_size * seq_len, num_experts, capacity)
    and combine weight (batch_size * seq_len, num_experts, capacity) for routing usage. More detailed
    function can be found in the paper about ViT-MoE.

    Args:
        capacity_factor_train (float, optional): Capacity factor in routing of training.
        capacity_factor_eval (float, optional): Capacity factor in routing of evaluation.
        min_capacity (int, optional): The minimum number of the capacity of each expert
        noisy_func (:class:`typing.Callable`, optional): Noisy function used in logits.
        drop_tks (bool, optional): Whether drops tokens in evaluation.
    """

    def __init__(self,
                 capacity_factor_train: float = 1.25,
                 capacity_factor_eval: float = 2.0,
                 min_capacity: int = 4,
                 noisy_func: Optional[Callable] = None,
                 drop_tks: bool = True):
        super().__init__(k_value=2,
                         capacity_factor_train=capacity_factor_train,
                         capacity_factor_eval=capacity_factor_eval,
                         min_capacity=min_capacity,
                         noisy_func=noisy_func,
                         drop_tks=drop_tks)

    def forward(self, inputs: torch.Tensor, use_kernel: bool = False, ep_group: Optional[ProcessGroup] = None) -> Tuple:
        """
        Args:
            inputs (torch.Tensor): The input tensor of shape (batch_size * seq_len, num_experts).

        Returns:
            1. use_kernel is False:
                The combine weight tensor of shape (batch_size * seq_len, num_experts, capacity).
                The dispatch mask tensor of shape (batch_size * seq_len, num_experts, capacity).
            2. use_kernel is True:
                ...
        """
        if self.noisy_func is not None and self.training:
            inputs = self.noisy_func(inputs)

        assert inputs.dtype == torch.float
        probs = F.softmax(inputs, dim=-1)
        num_experts = probs.size(-1)
        capacity = self.get_capacity(inputs.shape)

        top1_idx = torch.argmax(probs, dim=-1)
        mask1 = F.one_hot(top1_idx, num_classes=num_experts).to(torch.int32)
        logits_except1 = probs.masked_fill(mask1.bool(), float("-inf"))
        top2_idx = torch.argmax(logits_except1, dim=-1)
        mask2 = F.one_hot(top2_idx, num_classes=num_experts).to(torch.int32)

        cmask = (mask1 + mask2)    # loss: [s, e]
        cmask = cmask.float() / 2.0    # div 2 to normalize it to 1

        # calculate loss
        expert_indices = torch.stack([top1_idx, top2_idx], dim=-1)
        self.set_aux_loss(probs, expert_indices, num_experts)
        self.set_z_loss(inputs)
        self.pop_router_loss()

        if not self.training and not self.drop_tks and ep_group is not None:
            max_num = torch.max(torch.sum(cmask, dim=0))
            dist.all_reduce(max_num, op=dist.ReduceOp.MAX, group=ep_group)
            capacity = max_num.item()

        rank1 = moe_cumsum(mask1, use_kernel=self.use_kernel)    # rank1: [s, e]
        # print('mask1: ', mask1.shape, 'rank1: ', rank1.shape)
        rank2 = moe_cumsum(mask2, use_kernel=self.use_kernel)
        rank2 += torch.sum(mask1, dim=-2, keepdim=True)

        mask1 *= torch.lt(rank1, capacity)
        mask2 *= torch.lt(rank2, capacity)
        used_capacity = mask1.sum(dim=0) + mask2.sum(dim=0)
        # print(used_capacity, used_capacity.shape)

        rank1 = torch.sum(mask1 * rank1, dim=-1)
        rank2 = torch.sum(mask2 * rank2, dim=-1)

        if use_kernel:
            mask1 = torch.sum(mask1, dim=-1)
            mask2 = torch.sum(mask2, dim=-1)

            mask = torch.stack([mask1, mask2], dim=0).to(torch.int32)
            dest_idx = torch.stack([top1_idx * capacity + rank1, top2_idx * capacity + rank2], dim=0).to(torch.int32)

            return used_capacity, probs, mask, dest_idx, num_experts * capacity
        else:
            """
            The following code is equivalent to:

                ```
                weight1 = mask1 * probs.type_as(inputs)
                weight2 = mask2 * probs.type_as(inputs)
                rank1_sc = F.one_hot(rank1, num_classes=capacity)
                rank2_sc = F.one_hot(rank2, num_classes=capacity)

                cb_weight1 = weight1.unsqueeze(2) * rank1_sc.unsqueeze(1)
                cb_weight2 = weight2.unsqueeze(2) * rank2_sc.unsqueeze(1)
                cb_weight = cb_weight1 + cb_weight2
                sec_mask = cb_weight.bool()
                ```
            """
            weight1 = mask1 * probs.type_as(inputs)
            weight2 = mask2 * probs.type_as(inputs)
            
            mask = torch.cat([torch.unsqueeze(mask1, 1), torch.unsqueeze(mask2, 1)], dim=1).permute(2, 1, 0)
            weight = weight1 + weight2
            # print("mask: ", mask.shape)

            top_x_list = []
            weight_list = []
            for i in range(num_experts):
                idx, top_x = torch.where(mask[i])
                top_x_list.append(top_x)
                weight_list.append(weight[top_x, i].unsqueeze(1))
                
                # print("idx: ", idx, "top_x: ", top_x)
                # print(weight[top_x, i].unsqueeze(1).shape)
            
            return used_capacity, weight_list, top_x_list

            weight1 = mask1 * probs.type_as(inputs)
            weight2 = mask2 * probs.type_as(inputs)

            cb_weight = torch.zeros(inputs.shape + (capacity,), device=inputs.device)
            sec_mask = torch.zeros_like(cb_weight, dtype=torch.bool)
            indices = torch.arange(0, inputs.shape[0], device=inputs.device)
            cb_weight[indices, top1_idx[indices], rank1[indices]] += weight1[indices, top1_idx[indices]]
            cb_weight[indices, top2_idx[indices], rank2[indices]] += weight2[indices, top2_idx[indices]]
            sec_mask[indices, top1_idx[indices], rank1[indices]] |= mask1.bool()[indices, top1_idx[indices]]
            sec_mask[indices, top2_idx[indices], rank2[indices]] |= mask2.bool()[indices, top2_idx[indices]]

            return used_capacity, cb_weight, sec_mask