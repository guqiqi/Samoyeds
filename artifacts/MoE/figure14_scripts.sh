#!/bin/bash

gpu_id=${CUDA_VISIBLE_DEVICES:-0}
gpu_model=$(nvidia-smi -i $gpu_id --query-gpu=gpu_name --format=csv,noheader,nounits)
gpu_model=${gpu_model// /.}
cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)

SSMM_python_files=(mixtral_Samoyeds.py deepseek_Samoyeds.py)
MEGABLOCKS_python_files=(mixtral_megablocks.py deepseek_megablocks.py)
GEMM_python_files=(mixtral_transformers.py deepseek_transformers.py)
vLLM_python_files=(mixtral_vllm.py deepseek_vllm.py)

echo "GPU model: $gpu_model"
echo "CUDA version: $cuda_version"

output_dir="artifacts/results"
mkdir -p $output_dir

# batch_size, num_experts, intermediate_sizes, hidden_sizes, topk
# Mixtral defalut config is 1, 8, 14336, 4096, 2
# Mixtral Large default config is 1, 8, 16384, 6144, 2
# qwen default config is 1, 60, 1408, 2048, 4
# MiniCPM-MoE default config is 1, 8, 2304, 5760, 2
# DeepSeek default config is 1, 64, 1408, 2048, 6
# OpenMoE-34B default config is 1, 32, 12288, 3072, 2
config="1,8,14336,4096,4096 \
1,8,16384,6144,4096 \
1,8,5760,2304,4096 \
1,32,12288,3072,4096 \
16,64,1408,2048,4096 \
14,60,1408,2048,4096"

OLD_IFS="$IFS"

count=0
total=$(echo -e "$config" | awk -v RS=' ' 'END {print NR}')

start_string="model, model type, kernel type, iter, batch_size, seq_len, hidden_size, intermediate_size, expert_num, time"

output_file="$output_dir/MoE.csv"
echo $start_string > "${output_file}"

IFS=" "
for cfg in $config; do
    IFS=","; set -- $cfg
    batch=$1; expert=$2; intermediate_size=$3; hidden_size=$4; seq_len=$5

    ((count++))
    echo "Progress: ${count}/${total} (Running with batch_size=${batch} num_experts=${expert} intermediate_size=${intermediate_size} hidden_size=${hidden_size} seq_len=${seq_len})"

    echo "running Samoyeds moe"
    for file in "${SSMM_python_files[@]}"; do
        python $file --time --batch_size $batch --mlp --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size --seq_len $seq_len>> $output_file
    done

    echo "running megablocks moe"
    for file in "${MEGABLOCKS_python_files[@]}"; do
        python $file --time --batch_size $batch --mlp --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size --seq_len $seq_len>> $output_file
    done

    echo "running gemm moe"
    for file in "${GEMM_python_files[@]}"; do
        python $file --time --batch_size $batch --mlp --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size --seq_len $seq_len>> $output_file
    done

    echo "running vllm moe"
    for file in "${vLLM_python_files[@]}"; do
        python $file --time --batch_size $batch --mlp --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size --seq_len $seq_len>> $output_file
    done

    echo "done"
done

IFS="$OLD_IFS"
