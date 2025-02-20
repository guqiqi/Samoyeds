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

from MiniCPM.modeling_MiniCPM_samoyeds import SSMiniCPMModel, SSMiniCPMDecoderLayer, SSMiniCPMMoE
from MiniCPM.configuration_MiniCPM import MiniCPMConfig

parser = argparse.ArgumentParser()
parser.add_argument('--profile', action='store_true', default=False)
parser.add_argument('--time', action='store_true', default=False)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=4096)

parser.add_argument('--mlp', action='store_true', default=False)
parser.add_argument('--layer', action='store_true', default=False)
parser.add_argument('--model', action='store_true', default=False)

parser.add_argument('--hidden_size', type=int, default=4096)
parser.add_argument('--intermediate_size', type=int, default=11008)

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

configuration = MiniCPMConfig(
        vocab_size=32000, # default 32000
        hidden_size=k, # default 4096
        intermediate_size=m, # default 11008
        num_hidden_layers=1, # default 32
        num_attention_heads=32, # default 32
        num_key_value_heads=None, # default None
        hidden_act="silu", # default "silu"
        max_position_embeddings=2048, # default 2048
        initializer_range=0.02, # default 0.02
        rms_norm_eps=1e-6, # default 1e-6
        use_cache=True, # default True
        pad_token_id=None, # default None
        bos_token_id=1, # default 1
        eos_token_id=2, # default 2
        pretraining_tp=1, # default 1
        tie_word_embeddings=True, # default True
        rope_theta=10000.0, # default 10000.0
        rope_scaling=None, # default None
        attention_bias=False, # default False
        attention_dropout=0.0, # default 0.0
        scale_emb=1, # default 1
        dim_model_base=256, # default 256
        scale_depth=1.4, # default 1.4
        num_experts=expert_num, # default 8
        num_experts_per_tok=2, # default 0
    )

position_ids = None
if use_flash:
    configuration._attn_implementation = "flash_attention_2"
    position_ids = torch.arange(args.seq_len).unsqueeze(0).expand(args.batch_size, args.seq_len).cuda()
else:
    configuration._attn_implementation = "eager"

def MiniCPM_mlp_run():
    ss_model = SSMiniCPMMoE(configuration).half().cuda()
    ss_model.eval()

    # input形状为(batch_size, sequence_length, hidden_size)
    input = torch.rand((args.batch_size, args.seq_len, k)).half().cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                torch.cuda.synchronize()
                start = time.time()
            output = ss_model(input)
        torch.cuda.synchronize()
        end = time.time()
        print("MiniCPM,mlp,Samoyeds,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000, configuration._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './outputs/profiler/MiniCPM_Samoyeds_MLP_' + torch.cuda.get_device_name().split(' ')[1].split('-')[
                    0] + '_' + str(args.batch_size)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

        for i in range(ITER):
            # 执行过程
            output = ss_model(input)
            prof.step()

        prof.stop()

    pass


def MiniCPM_decoder_layer_run():
    ss_model = SSMiniCPMDecoderLayer(configuration, 0).half().cuda()
    ss_model.eval()

    # input形状为(batch_size, sequence_length, hidden_size)
    input = torch.rand((args.batch_size, args.seq_len, k)).half().cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                start = time.time()
            output = ss_model(input, position_ids=position_ids)
        torch.cuda.synchronize()
        end = time.time()
        print("MiniCPM,layer,Samoyeds,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000, configuration._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './outputs/profiler/MiniCPM_Samoyeds_Layer_' + torch.cuda.get_device_name().split(' ')[1].split('-')[
                    0] + '_' + str(args.batch_size)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

        for i in range(ITER):
            # 执行过程
            output = ss_model(input, position_ids=position_ids)
            prof.step()

        prof.stop()


def MiniCPM_model_run():
    # ================= 模型的整体替换 =================
    configuration.num_hidden_layers = 40
    ss_model = SSMiniCPMModel(configuration).half().cuda()
    ss_model.eval()

    # input形状为(batch_size, sequence_length)
    input = torch.randint(low=0, high=122753, size=(args.batch_size, args.seq_len)).cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                start = time.time()
            output = ss_model(input)
        end = time.time()
        print("MiniCPM,model,Samoyeds,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000, configuration._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=3, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './outputs/profiler/MiniCPM_Samoyeds_model_' + torch.cuda.get_device_name().split(' ')[1].split('-')[0]),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

        for i in range(ITER):
            # 执行过程
            output = ss_model(input)
            prof.step()

        prof.stop()


if __name__ == "__main__":
    # 设置随机数种子
    torch.manual_seed(123)

    # 对于CUDA，还需要设置随机数种子
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    torch.set_grad_enabled(False)

    # print('model,model type,kernel type,iter,batch_size,seq_len,hidden_size,intermediate_size,expert_num,time,atten_mode')

    if args.mlp:
        MiniCPM_mlp_run()
    if args.layer:
        MiniCPM_decoder_layer_run()
    if args.model:
        MiniCPM_model_run()
