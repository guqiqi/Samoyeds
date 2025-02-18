import argparse
import time

import torch

from qwen2_moe.modeling_qwen2_moe import Qwen2MoeModel, Qwen2MoeDecoderLayer, Qwen2MoeSparseMoeBlock, Qwen2MoeMLP
from qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig

parser = argparse.ArgumentParser()
parser.add_argument('--profile', action='store_true', default=False)
parser.add_argument('--time', action='store_true', default=False)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=4096)

parser.add_argument('--attention', action='store_true', default=False)
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

position_ids = None
if use_flash:
    configuration._attn_implementation = "flash_attention_2"
    position_ids = torch.arange(args.seq_len).unsqueeze(0).expand(args.batch_size, args.seq_len).cuda()
else:
    configuration._attn_implementation = "eager"

def attention_run():
    dense_model = Qwen2MoeDecoderLayer(configuration, 0).half()
    attention = dense_model.self_attn
    attention = attention.cuda()
    
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
        print("qwen2_moe,attention,GEMM,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000, configuration._attn_implementation))
        
def qwen2_moe_mlp_run():
    ss_model = Qwen2MoeSparseMoeBlock(configuration).half().cuda()
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
        print("qwen2_moe,mlp,GEMM,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start)*1000, configuration._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./outputs/profiler/qwen2_moe_GEMM_MLP_'+torch.cuda.get_device_name().split(' ')[1].split('-')[0]+'_'+str(args.batch_size)),
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
    ss_model = Qwen2MoeDecoderLayer(configuration, 0).half().cuda()
    ss_model.eval()

    input = torch.rand((args.batch_size, args.seq_len, k)).half().cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                torch.cuda.synchronize()
                start = time.time()
            output,  = ss_model(input, position_ids = position_ids)
        torch.cuda.synchronize()
        end = time.time()
        print("qwen2_moe,layer,GEMM,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start)*1000, configuration._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./outputs/profiler/qwen2_moe_GEMM_Layer_'+torch.cuda.get_device_name().split(' ')[1].split('-')[0]+'_'+str(args.batch_size)),
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
    ss_model = Qwen2MoeModel(configuration).half().cuda()
    ss_model.eval()

    # input形状为(batch_size, sequence_length)
    input = torch.randint(low=0, high=151936, size=(args.batch_size, args.seq_len)).cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                start = time.time()
            output = ss_model(input)
        end = time.time()
        print("qwen2_moe,model,GEMM,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start)*1000, configuration._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=3, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./outputs/profiler/qwen2_moe_GEMM_model_'+torch.cuda.get_device_name().split(' ')[1].split('-')[0]),
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

    # print('model,model type,kernel type,iter,batch_size,seq_len,hidden_size,intermediate_size,expert_num,time,atten_mode')
    
    if args.attention:
        attention_run()
    if args.mlp:
        qwen2_moe_mlp_run()
    if args.layer:
        qwen2_moe_decoder_layer_run()
    if args.model:
        qwen2_moe_model_run()
