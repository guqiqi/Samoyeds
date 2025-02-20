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

from module.moeblock.MoEBlock import MoEBlock
from module.moeblock.SPMoEBlock import SPMoEBlock
from module.moeblock.SSDenseMoEBlock import SSDenseMoEBlock
from module.moeblock.SSDenseTransMoEBlock import SSDenseTransMoEBlock
from module.moeblock.SSDenseFuseTransMoEBlock import SSDenseFuseTransMoEBlock
from module.moeblock.SSSparseFuseTransMoEBlock import SSSparseFuseTransMoEBlock

parser = argparse.ArgumentParser()
parser.add_argument('--time', action='store_true', default=False)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=4096)

parser.add_argument('--hidden_size', type=int, default=4096)
parser.add_argument('--intermediate_size', type=int, default=14336)

parser.add_argument('--experts', type=int, default=8)

args = parser.parse_args()

WARMUP = 10
ITER = 100

if __name__ == "__main__":
    # 设置随机数种子
    torch.manual_seed(2024)

    # 对于CUDA，还需要设置随机数种子
    torch.cuda.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    
    dense_model = MoEBlock(num_experts=args.experts, top_k=2, ffn_dim=args.hidden_size, hidden_dim=args.intermediate_size).half().cuda()
    dense_model.eval()
    
    input = torch.rand((args.batch_size, args.seq_len, args.hidden_size)).half().cuda()
    output = dense_model(input)
    
    if args.time:
        # print("Model,Iteration,batch_size,seq_len,hidden_size,intermediate_size,experts,Time(ms)")
        
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                start = time.time()
            output = dense_model(input)
        end = time.time()
        print("%s,%d,%d,%d,%d,%d,%d,%s" %
            (dense_model.__class__.__name__, ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000))
    
    model_list = [SPMoEBlock, SSDenseMoEBlock, SSDenseTransMoEBlock, SSDenseFuseTransMoEBlock, SSSparseFuseTransMoEBlock]
    # model_list = [SPMoEBlock, SSDenseMoEBlock]
    dense_model = dense_model.cpu()
    for model_name in model_list:
        model = model_name(num_experts=args.experts, top_k=2, ffn_dim=args.hidden_size, hidden_dim=args.intermediate_size).half().cuda()
        model.eval()
        
        output = model(input)
        
        if args.time:
            for i in range(ITER + WARMUP):
                if i == WARMUP:
                    start = time.time()
                output = model(input)
            end = time.time()
            print("%s,%d,%d,%d,%d,%d,%d,%s" %
                (model.__class__.__name__, ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, args.experts, (end - start) * 1000))
        