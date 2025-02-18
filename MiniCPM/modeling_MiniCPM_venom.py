# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch MiniCPM model."""
import math
import warnings
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
import re

from .configuration_MiniCPM import MiniCPMConfig
from .modeling_MiniCPM import MINICPM_ATTENTION_CLASSES, MiniCPMRMSNorm, MiniCPMPreTrainedModel, MINICPM_INPUTS_DOCSTRING

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
except:
    pass


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

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

class SPMiniCPMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.w1 = SrnmSpmm(nn.Linear(self.hidden_size, self.intermediate_size, bias=False))
        self.w3 = SrnmSpmm(nn.Linear(self.hidden_size, self.intermediate_size, bias=False))
        self.w2 = SrnmSpmm(nn.Linear(self.intermediate_size, self.hidden_size, bias=False))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        hidden_states = padding(hidden_states, 64)
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states[0:batch_size, :]


class SPMiniCPMMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList(
            [SPMiniCPMMLP(config) for i in range(self.num_experts)]
        )
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.intermediate_size = config.intermediate_size
    
    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(-1, orig_shape[-1])
        token_num = hidden_states.shape[0]
        scores = self.gate(hidden_states)
        scores_prob = F.softmax(scores, dim=-1, dtype=torch.float32)
        expert_weights, expert_indices = torch.topk(scores_prob, self.num_experts_per_tok, dim=-1)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        topk_idx_flat = expert_indices.view(-1)
        expert_weights = expert_weights.to(orig_dtype)

        y = self.moe_infer(hidden_states, topk_idx_flat, expert_weights.view(-1, 1)).view(*orig_shape)
        return y
    
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache
    
class SPMiniCPMDecoderLayer(nn.Module):
    def __init__(self, config: MiniCPMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MINICPM_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = SPMiniCPMMoE(config)
        self.input_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
class SPMiniCPMModel(MiniCPMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MiniCPMDecoderLayer`]

    Args:
        config: MiniCPMConfig
    """

    def __init__(self, config: MiniCPMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SPMiniCPMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )