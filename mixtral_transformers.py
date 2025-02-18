import argparse
import time

import torch

from mixtral.modeling_mixtral import MixtralModel, MixtralSparseMoeBlock, MixtralBLockSparseTop2MLP, MixtralDecoderLayer
from mixtral.configuration_mixtral import MixtralConfig

parser = argparse.ArgumentParser()
parser.add_argument('--profile', action='store_true', default=False)
parser.add_argument('--time', action='store_true', default=False)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=4096)

parser.add_argument('--attention', action='store_true', default=False)
parser.add_argument('--mlp', action='store_true', default=False)
parser.add_argument('--layer', action='store_true', default=False)
parser.add_argument('--model', action='store_true', default=False)

parser.add_argument('--hidden_size', type=int, default=4096)
parser.add_argument('--intermediate_size', type=int, default=14336)

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

# setup Mixtral configuration
configuration = MixtralConfig(
    vocab_size=32000,  # default 32000
    hidden_size=k,  # default 4096
    intermediate_size=m,  # default 14336
    num_hidden_layers=1,  # default 32
    num_attention_heads=32,  # default 32
    num_key_value_heads=8,  # default 8
    hidden_act="silu",  # default "silu"
    max_position_embeddings=4096 * 32,  # default 4096 * 32
    initializer_range=0.02,  # default 0.02
    rms_norm_eps=1e-5,  # default 1e-5
    use_cache=True,  # default True
    pad_token_id=None,  # default None
    bos_token_id=1,  # default 1
    eos_token_id=2,  # default 2
    tie_word_embeddings=False,  # default False
    rope_theta=1e6,  # default 1e6
    sliding_window=4096,  # default 4096
    attention_dropout=0.0,  # default 0.0
    num_experts_per_tok=2,  # default 2
    num_local_experts=expert_num,  # default 8
    output_router_logits=False,  # default False
    router_aux_loss_coef=0.001,  # default 0.001
)

position_ids = None
if use_flash:
    configuration._attn_implementation = "flash_attention_2"
    position_ids = torch.arange(args.seq_len).unsqueeze(0).expand(args.batch_size, args.seq_len).cuda()
else:
    configuration._attn_implementation = "eager"

def attention_run():
    dense_model = MixtralDecoderLayer(configuration, 0).half()
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
        print("Mixtral,attention,GEMM,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000, configuration._attn_implementation))

def mixtral_mlp_run():
    # ================= MixtralSparseMoeBlock的替换 =================
    dense_model = MixtralSparseMoeBlock(configuration)
    dense_model = dense_model.half().cuda()
    dense_model.eval()
    # print("Aftering loading MixtralSparseMoeBlock...")

    # input形状为(batch_size, sequence_length, hidden_size)
    input = torch.rand((args.batch_size, args.seq_len, k)).half().cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                torch.cuda.synchronize()
                start = time.time()
            final_hidden_states, router_logits = dense_model(input)
        torch.cuda.synchronize()
        end = time.time()
        print("Mixtral,mlp,GEMM,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000, configuration._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './outputs/profiler/Mixtral_GEMM_MLP_' + torch.cuda.get_device_name().split(' ')[1].split('-')[
                    0] + '_' + str(args.batch_size)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

        for i in range(ITER):
            # 执行过程
            final_hidden_states, router_logits = dense_model(input)
            prof.step()

        prof.stop()

    pass


def mixtral_decoder_layer_run():
    # ================= MixtralDecoderLayer的替换 =================
    dense_model = MixtralDecoderLayer(configuration, 0).half().cuda()
    dense_model.eval()
    # print("Aftering loading MixtralDecoderLayer...")

    # input形状为(batch_size, sequence_length, hidden_size)
    input = torch.rand((args.batch_size, args.seq_len, k)).half().cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                start = time.time()
            output, = dense_model(input, position_ids = position_ids)
        torch.cuda.synchronize()
        end = time.time()
        print("Mixtral,layer,GEMM,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000, configuration._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './outputs/profiler/Mixtral_GEMM_Layer_' + torch.cuda.get_device_name().split(' ')[1].split('-')[
                    0] + '_' + str(args.batch_size)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

        for i in range(ITER):
            # 执行过程
            output, = dense_model(input)
            prof.step()

        prof.stop()


def mixtral_model_run():
    # ================= 模型的整体替换 =================
    configuration.num_hidden_layers = 32
    dense_model = MixtralModel(configuration).half().cuda()
    dense_model.eval()
    # print("Aftering loading MixtralModel...")

    # input形状为(batch_size, sequence_length)
    input = torch.randint(low=0, high=32000, size=(args.batch_size, args.seq_len)).cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                start = time.time()
            output = dense_model(input)
        end = time.time()
        print("Mixtral,model,GEMM,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, (args.experts, end - start) * 1000, configuration._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=3, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './outputs/profiler/Mixtral_GEMM_model_' + torch.cuda.get_device_name().split(' ')[1].split('-')[0]),
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

    if args.attention:
        attention_run()
    if args.mlp:
        mixtral_mlp_run()
    if args.layer:
        mixtral_decoder_layer_run()
    if args.model:
        mixtral_model_run()
