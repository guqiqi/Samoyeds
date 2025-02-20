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

import argparse
import time

import torch

from openmoe.modeling_openmoe import OpenMoeDecoderLayer
from colossalai.moe.layers import SparseMLP

from transformers.models.llama.configuration_llama import LlamaConfig


parser = argparse.ArgumentParser()
parser.add_argument('--profile', action='store_true', default=False)
parser.add_argument('--time', action='store_true', default=False)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=2048)

parser.add_argument('--attention', action='store_true', default=False)
parser.add_argument('--mlp', action='store_true', default=False)
parser.add_argument('--layer', action='store_true', default=False)
parser.add_argument('--model', action='store_true', default=False)

parser.add_argument('--hidden_size', type=int, default=2048)
parser.add_argument('--intermediate_size', type=int, default=8192)

parser.add_argument('--experts', type=int, default=32)

parser.add_argument('--flash', action='store_true', default=False)

args = parser.parse_args()

m = args.intermediate_size
k = args.hidden_size
n = args.batch_size * args.seq_len
expert_num = args.experts
use_flash = args.flash

WARMUP = 10
ITER = 100

# setup LlamaConfig configuration
config = LlamaConfig(
    vocab_size=256384,  # default 256384
    hidden_size=k,  # default 2048
    intermediate_size=m,  # default 8192
    num_hidden_layers=1,  # default 24
    num_attention_heads=24,  # default 24
    num_key_value_heads=24,  # default 24
    hidden_act="swiglu",  # default "swiglu"
    max_position_embeddings=2048,  # default 2048
    initializer_range=0.02,  # default 0.02
    rms_norm_eps=1e-6,  # default 1e-6
    use_cache=True,  # default True
    pad_token_id=None,  # default None
    bos_token_id=1,  # default 1
    eos_token_id=2,  # default 2
    tie_word_embeddings=False,  # default False
    rope_theta=1e6,  # default 1e6
    sliding_window=4096,  # default 4096
    attention_dropout=0.0,  # default 0.0
    router_topk=2,  # default 2
    num_experts=expert_num,  # default 32
    moe_layer_interval=6,  # default 6
    output_router_logits=False,  # default False
    router_aux_loss_coef=0.001,  # default 0.001
    dropout_rate=0.0, # default 0.0
    head_dim=128, # default 128
    router_capacity_factor_train=1.25,
    router_capacity_factor_eval=2.0,
    router_min_capacity=4,
    router_noisy_policy=None,
    router_drop_tks=True,
    router_aux_loss_factor=0.01,
    router_z_loss_factor=0.001,
    mlp_gated=True,
    label_smoothing=0.001,
    z_loss_factor=0.01,
    enable_load_balance=False,
    load_balance_tolerance=0.1,
    load_balance_beam_width=8,
    load_balance_group_swap_factor=0.4,
    enable_kernel=False,
    enable_comm_overlap=False,
    enable_hierarchical_alltoall=False,
    pretraining_tp=1
)

position_ids = None
if use_flash:
    config._attn_implementation = "flash_attention_2"
    position_ids = torch.arange(args.seq_len).unsqueeze(0).expand(args.batch_size, args.seq_len).cuda()
else:
    config._attn_implementation = "eager"

def attention_run():
    dense_model = OpenMoeDecoderLayer(config, moe=True, use_flash=use_flash)
    attention = dense_model.self_attn
    attention = attention.half().cuda()
    
    attention.eval()
    
    # input形状为(batch_size, sequence_length, hidden_size)
    input = torch.rand((args.batch_size, args.seq_len, k)).half().cuda()
    
    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                torch.cuda.synchronize()
                start = time.time()
            output = attention(input, position_ids=position_ids)
        torch.cuda.synchronize()
        end = time.time()
        print("OpenMoE,attention,GEMM,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000, config._attn_implementation))


def openmoe_mlp_run():
    # ================= OpenMoeMLP =================
    model = SparseMLP(num_experts=config.num_experts,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                router_top_k=config.router_topk,
                router_capacity_factor_train=config.router_capacity_factor_train,
                router_capacity_factor_eval=config.router_capacity_factor_eval,
                router_min_capacity=config.router_min_capacity,
                router_noisy_policy=config.router_noisy_policy,
                router_drop_tks=config.router_drop_tks,
                mlp_activation=config.hidden_act,
                mlp_gated=config.mlp_gated,
                enable_load_balance=config.enable_load_balance,
                load_balance_tolerance=config.load_balance_tolerance,
                load_balance_beam_width=config.load_balance_beam_width,
                load_balance_group_swap_factor=config.load_balance_group_swap_factor,
                enable_kernel=config.enable_kernel,
                enable_comm_overlap=config.enable_comm_overlap,)
    model = model.half().cuda()
    model.eval()

    # input形状为(batch_size, sequence_length, hidden_size)
    input = torch.rand((args.batch_size, args.seq_len, k)).half().cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                torch.cuda.synchronize()
                start = time.time()
            outputs = model(input)
        torch.cuda.synchronize()
        end = time.time()
        print("OpenMoE,mlp,GEMM,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000, config._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './outputs/profiler/OpenMoE_GEMM_MLP_' + torch.cuda.get_device_name().split(' ')[1].split('-')[
                    0] + '_' + str(args.batch_size)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

        for i in range(ITER):
            # 执行过程
            outputs = model(input)
            prof.step()

        prof.stop()
    pass

def openmoe_layer_run():
    model = OpenMoeDecoderLayer(config, moe=True, use_flash=use_flash)
    model = model.half().cuda()
    model.eval()
    
    # input形状为(batch, seq_len, embed_dim=hidden_size)
    input = torch.rand((args.batch_size, args.seq_len, k)).half().cuda()
    # outputs = model(input)

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                torch.cuda.synchronize()
                start = time.time()
            outputs = model(input, position_ids=position_ids)
        torch.cuda.synchronize()
        end = time.time()
        print("OpenMoE,layer,GEMM,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000, config._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './outputs/profiler/OpenMoE_GEMM_layer_' + torch.cuda.get_device_name().split(' ')[1].split('-')[
                    0] + '_' + str(args.batch_size)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

        for i in range(ITER):
            # 执行过程
            outputs = model(input, position_ids=position_ids)
            prof.step()

        prof.stop()
    pass

if __name__ == "__main__":
    # 设置随机数种子
    torch.manual_seed(123)

    # 对于CUDA，还需要设置随机数种子
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    torch.set_grad_enabled(False)

    # print('model, model type, kernel type, iter, batch_size, seq_len, hidden_size, intermediate_size, expert_num, time')

    if args.attention:
        attention_run()
    if args.mlp:
        openmoe_mlp_run()
    if args.layer:
        openmoe_layer_run()
    if args.model:
        pass