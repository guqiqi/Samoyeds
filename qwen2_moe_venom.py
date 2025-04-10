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

from qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig

from qwen2_moe.modeling_qwen2_moe_venom import SPQwen2MoeSparseMoeBlock, SPQwen2MoeDecoderLayer, SPQwen2MoeModel

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

args = parser.parse_args()

m = args.intermediate_size
k = args.hidden_size
n = args.batch_size * args.seq_len
expert_num = args.experts

shared_intermediate_size = 5632

WARMUP = 10
ITER = 100

configuration = Qwen2MoeConfig(
        vocab_size=151936, # default 151936
        hidden_size=k, # default 2048
        intermediate_size=shared_intermediate_size, # default 5632
        num_hidden_layers=1, # default 24
        num_attention_heads=16, # default 16
        num_key_value_heads=16, # default 16
        hidden_act="silu", # default "silu"
        max_position_embeddings=32768, # default 32768
        initializer_range=0.02, # default 0.02
        rms_norm_eps=1e-6, # default 1e-6
        use_cache=True, # default True
        tie_word_embeddings=False, # default False
        rope_theta=10000.0, # default 10000.0
        use_sliding_window=False, # default False
        sliding_window=4096, # default 4096
        max_window_layers=28, # default 28
        attention_dropout=0.0, # default 0.0
        decoder_sparse_step=1, # default 1
        moe_intermediate_size=m, # default 1408
        shared_expert_intermediate_size=shared_intermediate_size, # default 5632
        num_experts_per_tok=4, # default 4
        num_experts=expert_num, # default 60
        norm_topk_prob=False, # default False
        output_router_logits=False, # default False
        router_aux_loss_coef=0.001, # default 0.001
    )

def qwen2_moe_mlp_run():
    ss_model = SPQwen2MoeSparseMoeBlock(configuration).half().cuda()
    ss_model.eval()

    input = torch.rand((args.batch_size, args.seq_len, k)).half().cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                torch.cuda.synchronize()
                start = time.time()
            final_hidden_states, router_logits = ss_model(input)
        torch.cuda.synchronize()
        end = time.time()
        print("qwen2_moe, mlp, VENOM, %d, %d, %d, %d, %d, %d, %s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start)*1000))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./outputs/profiler/qwen2_moe_VENOM_MLP_'+torch.cuda.get_device_name().split(' ')[1].split('-')[0]+'_'+str(args.batch_size)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

        for i in range(ITER):
            # 执行过程
            final_hidden_states, router_logits = ss_model(input)
            prof.step()

        prof.stop()

    pass

def qwen2_moe_decoder_layer_run():
    ss_model = SPQwen2MoeDecoderLayer(configuration, 0).half().cuda()
    ss_model.eval()

    input = torch.rand((args.batch_size, args.seq_len, k)).half().cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                torch.cuda.synchronize()
                start = time.time()
            output,  = ss_model(input)
        torch.cuda.synchronize()
        end = time.time()
        print("qwen2_moe, layer, VENOM, %d, %d, %d, %d, %d, %d, %s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start)*1000))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./outputs/profiler/qwen2_moe_VENOM_Layer_'+torch.cuda.get_device_name().split(' ')[1].split('-')[0]+'_'+str(args.batch_size)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

        for i in range(ITER):
            # 执行过程
            output,  = ss_model(input)
            prof.step()

        prof.stop()

def qwen2_moe_model_run():
    configuration.num_hidden_layers = 24
    ss_model = SPQwen2MoeModel(configuration).half().cuda()
    ss_model.eval()

    # input形状为(batch_size, sequence_length)
    input = torch.randint(low=0, high=151936, size=(args.batch_size, args.seq_len)).cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                start = time.time()
            output = ss_model(input)
        end = time.time()
        print("qwen2_moe, model, VENOM, %d, %d, %d, %d, %d, %d, %s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start)*1000))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=3, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./outputs/profiler/qwen2_moe_VENOM_model_'+torch.cuda.get_device_name().split(' ')[1].split('-')[0]),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

        for i in range(ITER):
            output  = ss_model(input)
            prof.step()

        prof.stop()

if __name__ == "__main__":
    # 设置随机数种子
    torch.manual_seed(123)

    # 对于CUDA，还需要设置随机数种子
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    torch.set_grad_enabled(False)

    # print('model, model type, kernel type, iter, batch_size, seq_len, hidden_size, intermediate_size, expert_num, time')

    if args.mlp:
        qwen2_moe_mlp_run()
    if args.layer:
        qwen2_moe_decoder_layer_run()
    if args.model:
        qwen2_moe_model_run()
