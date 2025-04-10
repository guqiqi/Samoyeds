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

from megablocks.layers.arguments import Arguments
from megablocks.layers import dmoe

import argparse
import time
import os

import torch
import torch.distributed
import torch.nn as nn

from deepseek.configuration_deepseek import DeepseekConfig
from vllm.model_executor.parallel_utils.parallel_state import initialize_model_parallel
# from vllm.distributed.parallel_state import initialize_model_parallel

from deepseek.modeling_deepseek_megablocks import DeepSeekBlockSparseMoE
from deepseek.modeling_deepseek_vllm import DeepseekDecoderLayer

parser = argparse.ArgumentParser()
parser.add_argument('--profile', action='store_true', default=False)
parser.add_argument('--time', action='store_true', default=False)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=4096)

parser.add_argument('--mlp', action='store_true', default=False)
parser.add_argument('--layer', action='store_true', default=False)
parser.add_argument('--model', action='store_true', default=False)

parser.add_argument('--hidden_size', type=int, default=2048)
parser.add_argument('--intermediate_size', type=int, default=1408)

parser.add_argument('--experts', type=int, default=8)

parser.add_argument('--flash', action='store_true', default=False)

args = parser.parse_args()

m = args.intermediate_size
k = args.hidden_size
n = args.batch_size * args.seq_len
expert_num = args.experts
use_flash = args.flash

WARMUP = 10
ITER = 100
shared_intermediate_size = 10944

configuration = DeepseekConfig(
        vocab_size=102400, # default 102400
        hidden_size=k, # default 2048
        intermediate_size=shared_intermediate_size, # default 10944
        moe_intermediate_size = m, # default 1408
        num_hidden_layers=1, # default 30
        num_attention_heads=16, # default 32
        num_key_value_heads=16, # default 32
        n_shared_experts = 2, # default None
        n_routed_experts = expert_num, # default 64
        num_experts_per_tok = 6, # default None
        moe_layer_freq = 1, # default 1
        first_k_dense_replace = 0, # default 0
        norm_topk_prob = False, # default False
        scoring_func = 'softmax', # default 'softmax'
        aux_loss_alpha = 0.001, # default 0.001
        seq_aux = True, # default True
        hidden_act="silu", # default "silu"
        max_position_embeddings=2048, # default 2048
        initializer_range=0.02, # default 0.02
        rms_norm_eps=1e-6, # default 1e-6
        use_cache=True, # default True
        pad_token_id=None, # default None
        bos_token_id=100000, # default 100000
        eos_token_id=100001, # default 100001
        pretraining_tp=1, # default 1
        tie_word_embeddings=False, # default False
        rope_theta=10000.0, # default 10000.0
        rope_scaling=None, # default None
        attention_bias=False, # default False
        attention_dropout=0.0, # default 0.0
    )

position_ids = None
if use_flash:
    configuration._attn_implementation = "flash_attention_2"
    position_ids = torch.arange(args.seq_len).unsqueeze(0).expand(args.batch_size, args.seq_len).cuda()
else:
    configuration._attn_implementation = "eager"

def deepseek_mlp_run():
    dmoe_mlp = DeepSeekBlockSparseMoE(
        num_experts=configuration.n_routed_experts,
        top_k=configuration.num_experts_per_tok,
        hidden_dim=configuration.hidden_size,
        ffn_dim=configuration.moe_intermediate_size,
        n_shared_experts=configuration.n_shared_experts).to(torch.bfloat16).cuda()
    dmoe_mlp.eval()

    # input = torch.rand((n, k)).half().cuda()

    # dmoe_mlp = dmoe.dMoE(moe_args)

    # dmoe_mlp.cuda(torch.cuda.current_device()).to(torch.bfloat16)
    
    # dmoe_mlp.eval()
    
    x = torch.randn(args.batch_size, args.seq_len, args.hidden_size).to(torch.bfloat16).cuda()

    out = dmoe_mlp(x)
    
    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                torch.cuda.synchronize()
                start = time.time()
            out = dmoe_mlp(x)
        torch.cuda.synchronize()
        end = time.time()
        print("DeepSeek,mlp,megablocks,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start)*1000, configuration._attn_implementation))

    pass

def deepseek_decoder_layer_run():
    ss_model = DeepseekDecoderLayer(configuration, 0)
    ss_model.mlp = DeepSeekBlockSparseMoE(
        num_experts=configuration.n_routed_experts,
        top_k=configuration.num_experts_per_tok,
        hidden_dim=configuration.hidden_size,
        ffn_dim=configuration.moe_intermediate_size,
        n_shared_experts=configuration.n_shared_experts)
    ss_model = ss_model.to(torch.bfloat16).cuda()
    ss_model.eval()

    input = torch.rand((args.batch_size, args.seq_len, k)).to(torch.bfloat16).cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                start = time.time()
            output = ss_model(input, position_ids=position_ids)
        torch.cuda.synchronize()
        end = time.time()
        print("DeepSeek,layer,megablocks,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000, configuration._attn_implementation))

if __name__ == "__main__":
    # 设置随机数种子
    torch.manual_seed(123)

    # 对于CUDA，还需要设置随机数种子
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    torch.set_grad_enabled(False)
    
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "9834"
    torch.distributed.init_process_group(world_size=1, rank=0)
    initialize_model_parallel()

    # print('model,model type,kernel type,iter,batch_size,seq_len,hidden_size,intermediate_size,expert_num,time,atten_mode')

    if args.mlp:
        deepseek_mlp_run()
    if args.layer:
        deepseek_decoder_layer_run()
    if args.model:
        pass

