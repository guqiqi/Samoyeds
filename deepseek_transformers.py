import argparse
import time

import torch

from deepseek.modeling_deepseek import DeepseekModel, DeepseekMoE, DeepseekDecoderLayer
from deepseek.configuration_deepseek import DeepseekConfig

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

shared_intermediate_size = 10944

WARMUP = 10
ITER = 100

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

def attention_run():
    dense_model = DeepseekDecoderLayer(configuration, moe=True, use_flash=use_flash)
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
        print("DeepSeek,attention,GEMM,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000, config._attn_implementation))

def deepseek_mlp_run():
    ss_model = DeepseekMoE(configuration).half().cuda()

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
        print("DeepSeek,mlp,GEMM,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000, configuration._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './outputs/profiler/DeepSeek_GEMM_MLP_' + torch.cuda.get_device_name().split(' ')[1].split('-')[
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


def deepseek_decoder_layer_run():
    ss_model = DeepseekDecoderLayer(configuration, 0).half().cuda()

    input = torch.rand((args.batch_size, args.seq_len, k)).half().cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                start = time.time()
            output = ss_model(input, position_ids = position_ids)
        torch.cuda.synchronize()
        end = time.time()
        print("DeepSeek,layer,GEMM,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000, configuration._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './outputs/profiler/DeepSeek_GEMM_Layer_' + torch.cuda.get_device_name().split(' ')[1].split('-')[
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


def deepseek_model_run():
    configuration.num_hidden_layers = 28
    dense_model = DeepseekModel(configuration).half().cuda()

    # input形状为(batch_size, sequence_length)
    input = torch.randint(low=0, high=102400, size=(args.batch_size, args.seq_len)).cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                start = time.time()
            output = dense_model(input)
        end = time.time()
        print("DeepSeek,model,GEMM,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, (end - start) * 1000, configuration._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=3, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './outputs/profiler/DeepSeek_GEMM_model_' + torch.cuda.get_device_name().split(' ')[1].split('-')[0]),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

        for i in range(ITER):
            # 执行过程
            output = dense_model(input)
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
        deepseek_mlp_run()
    if args.layer:
        deepseek_decoder_layer_run()
    if args.model:
        deepseek_model_run()
