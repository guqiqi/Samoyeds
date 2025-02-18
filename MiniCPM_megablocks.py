import argparse
import time
import os

import torch

from mixtral.modeling_mixtral_megablocks import BlockSparseMoE

from MiniCPM.modeling_MiniCPM_vllm import MiniCPMDecoderLayer
from MiniCPM.configuration_MiniCPM import MiniCPMConfig

from vllm.model_executor.parallel_utils.parallel_state import initialize_model_parallel

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
    dmoe_mlp = BlockSparseMoE(
        num_experts=configuration.num_experts,
        top_k=configuration.num_experts_per_tok,
        hidden_dim=configuration.hidden_size,
        ffn_dim=configuration.intermediate_size).to(torch.bfloat16).cuda()
    dmoe_mlp.eval()

    # input形状为(batch_size, sequence_length, hidden_size)
    input = torch.rand((args.batch_size, args.seq_len, k)).to(torch.bfloat16).cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                torch.cuda.synchronize()
                start = time.time()
            output = dmoe_mlp(input)
        torch.cuda.synchronize()
        end = time.time()
        print("MiniCPM,mlp,megablocks,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000, configuration._attn_implementation))


def MiniCPM_decoder_layer_run():
    model = MiniCPMDecoderLayer(configuration, 0)
    model.mlp = BlockSparseMoE(
        num_experts=configuration.num_experts,
        top_k=configuration.num_experts_per_tok,
        hidden_dim=configuration.hidden_size,
        ffn_dim=configuration.intermediate_size)
    model = model.to(torch.bfloat16).cuda()
    model.eval()

    # input形状为(batch_size, sequence_length, hidden_size)
    input = torch.rand((args.batch_size, args.seq_len, k)).to(torch.bfloat16).cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                start = time.time()
            output = model(input, position_ids = position_ids)
        torch.cuda.synchronize()
        end = time.time()
        print("MiniCPM,layer,megablocks,%d,%d,%d,%d,%d,%d,%s,%s" %
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
        MiniCPM_mlp_run()
    if args.layer:
        MiniCPM_decoder_layer_run()
    if args.model:
        pass