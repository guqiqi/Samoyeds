#!/bin/bash

gpu_id=${CUDA_VISIBLE_DEVICES:-0}
gpu_model=$(nvidia-smi -i $gpu_id --query-gpu=gpu_name --format=csv,noheader,nounits)
gpu_model=${gpu_model// /.}
cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)

echo "GPU model: $gpu_model"
echo "CUDA version: $cuda_version"

output_dir="artifacts/results"
mkdir -p $output_dir

# batch_size, num_experts, intermediate_sizes, hidden_sizes, topk, seq_len
# Mixtral defalut config is 1, 8, 14336, 4096, 2
# Mixtral Large default config is 1, 8, 16384, 6144, 2
# qwen default config is 1, 60, 1408, 2048, 4
# MiniCPM-MoE default config is 1, 8, 2304, 5760, 2
# MiniCPM default config is 1, 8, 4096, 11008, 2
# DeepSeek default config is 1, 64, 1408, 2048, 6
# OpenMoE-base(637M) default config is 1, 16, 2048, 768, 2
# OpenMoE-8B default config is 1, 32, 8192, 2048, 2
# OpenMoE-34B default config is 1, 32, 12288, 3072, 2

config="deepseek,56,64,1408,2048,4096 \
qwen2_moe,48,60,1408,2048,4096 \
MiniCPM,124,8,5760,2304,1024 \
openmoe,60,32,12288,3072,1024 \
mixtral,90,8,14336,4096,1024 \
mixtral,56,8,16384,6144,1024 \
"

OLD_IFS="$IFS"

count=0
total=$(echo -e "$config" | awk -v RS=' ' 'END {print NR}')

start_string="model,model type,kernel type,iter,batch_size,seq_len,hidden_size,intermediate_size,expert_num,time,atten_mode"

output_file="$output_dir/decoder_layer_batch_size.csv"
echo $start_string > "${output_file}"

IFS=" "
for cfg in $config; do
    IFS=","; set -- $cfg
    model=$1; batch_size=$2; expert=$3; intermediate_size=$4; hidden_size=$5; seq_len=$6

    for ((batch=1; batch<=batch_size; batch++))
    do
        ((count++))
        echo "Progress: ${count}/${total} (Running ${model} with batch_size=${batch} num_experts=${expert} intermediate_size=${intermediate_size} hidden_size=${hidden_size} seq_len=${seq_len})"

        echo "running Samoyeds moe"
        python ${model}_Samoyeds.py --time --flash --batch_size $batch --layer --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size --seq_len $seq_len>> $output_file

        echo "running gemm moe"
        python ${model}_transformers.py --time --flash --batch_size $batch --layer --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size --seq_len $seq_len>> $output_file

        echo "running vllm moe"
        python ${model}_vllm.py --time --flash --batch_size $batch --layer --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size --seq_len $seq_len>> $output_file

        echo "running megablocks moe"
        python ${model}_megablocks.py --time --flash --batch_size $batch --layer --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size --seq_len $seq_len>> $output_file
    done
    
    echo "done"
done

IFS="$OLD_IFS"
