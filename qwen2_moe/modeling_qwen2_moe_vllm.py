import warnings
from typing import List, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP, Qwen2MoeRMSNorm, QWEN2MOE_ATTENTION_CLASSES
from qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig

from vllm.model_executor.models.qwen2_moe import Qwen2MoeSparseMoeBlock

class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = QWEN2MOE_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        if config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0:
            self.mlp = Qwen2MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen2MoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

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
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.post_attention_layernorm(hidden_states)
        origin_shape = hidden_states.shape
        hidden_states = self.mlp(hidden_states.view(-1, self.hidden_size))
        hidden_states = hidden_states.reshape(origin_shape)
        
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs
