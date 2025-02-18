#!/bin/bash

gpu_id=${CUDA_VISIBLE_DEVICES:-0}
gpu_model=$(nvidia-smi -i $gpu_id --query-gpu=gpu_name --format=csv,noheader,nounits)
gpu_model=${gpu_model// /.}
cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)

echo "GPU model: $gpu_model"
echo "CUDA version: $cuda_version"

output_dir="artifacts/results"
mkdir -p $output_dir

# batch_size, num_experts, intermediate_sizes, hidden_sizes, topk
# Mixtral defalut config is 1, 8, 14336, 4096, 2
# Mixtral Large default config is 1, 8, 16384, 6144, 2
# qwen default config is 1, 60, 1408, 2048, 4
# MiniCPM-MoE default config is 1, 8, 2304, 5760, 2
# MiniCPM default config is 1, 8, 4096, 11008, 2
# DeepSeek default config is 1, 64, 1408, 2048, 6
# OpenMoE-base(637M) default config is 1, 16, 2048, 768, 2
# OpenMoE-8B default config is 1, 32, 8192, 2048, 2
# OpenMoE-34B default config is 1, 32, 12288, 3072, 2
config="2,8,14336,4096, \
1,8,16384,6144, \
8,8,5760,2304, \
2,32,12288,3072, \
16,64,1408,2048, \
16,60,1408,2048"

config="2,8,14336,4096, \
1,8,16384,6144, \
8,8,5760,2304, \
2,32,12288,3072, \
16,64,1408,2048, \
16,60,1408,2048"

OLD_IFS="$IFS"

count=0
config_lines=$(echo "$config" | tr '\' '\n')
total=$(echo "$config_lines" | grep -v '^$' | wc -l)

start_string="Model,Iteration,batch_size,seq_len,hidden_size,intermediate_size,experts,Time(ms)"

output_file="$output_dir/ablation.csv"
echo $start_string > "${output_file}"

IFS=" "
for cfg in $config; do
    IFS=","; set -- $cfg
    batch=$1; expert=$2; intermediate_size=$3; hidden_size=$4;

    ((count++))
    echo "Progress: ${count}/${total} (Running batch_size=${batch} num_experts=${expert} intermediate_size=${intermediate_size} hidden_size=${hidden_size})"

    python benchmark/benchmark_MoE.py --time --batch_size $batch --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size>> $output_file
    
    echo "done"
done

IFS="$OLD_IFS"
