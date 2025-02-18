#!/bin/bash

gpu_id=${CUDA_VISIBLE_DEVICES:-0}
gpu_model=$(nvidia-smi -i $gpu_id --query-gpu=gpu_name --format=csv,noheader,nounits)
gpu_model=${gpu_model// /.}
cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)

config="14336,4096,4096, \
16384,6144,4096, \
5760,2304,4096, \
8192,2048,4096, \
12288,3072,4096, \
1408,2048,4096"

output_dir="artifacts/results/kernel_model_config"
mkdir -p $output_dir

ssmm_output_file="$output_dir/SSMM_${gpu_model}_CUDA${cuda_version}.txt"
: >| $ssmm_output_file

sputnik_output_file="$output_dir/Sputnik_and_cuBlas_${gpu_model}_CUDA${cuda_version}.txt"
: >| $sputnik_output_file
echo "algo,arch,m,k,n,meta_block_sz,block_sz,nn_row,mm_col,density,bm,bn,bk,wm,wn,wk,mm,mn,mk,nstage,spmm_time,gemm_time,speedup,error" > $sputnik_output_file

venom_output_file="$output_dir/Venom_and_cuBlas_${gpu_model}_CUDA${cuda_version}.txt"
: >| $venom_output_file
echo "algo,arch,m,k,n,meta_block_sz,block_sz,nn_row,mm_col,density,bm,bn,bk,wm,wn,wk,mm,mn,mk,nstage,spmm_time,gemm_time,speedup,error" > $venom_output_file

cusparselt_output_file="$output_dir/cuSparseLtsearched_and_cuBlas_${gpu_model}_CUDA${cuda_version}.txt"
: >| $cusparselt_output_file
echo "algo,arch,m,k,n,meta_block_sz,block_sz,nn_row,mm_col,density,bm,bn,bk,wm,wn,wk,mm,mn,mk,nstage,spmm_time,gemm_time,speedup,error" > $cusparselt_output_file

for cfg in $config; do
  IFS=","; set -- $cfg
  m=$1; k=$2; n=$3;

  # SSMM
  echo "Running with m=$m, n=$n, k=$k" >> $ssmm_output_file
  ./Samoyeds-Kernel/build/benchmark/benchmark -m $m -n $n -k $k -N 1 -M 2 --vector_length 128 --method SSMM >> $ssmm_output_file

  # Sputnik
  ./Samoyeds-Kernel/benchmark/third_party/venom/build/src/benchmark_spmm --sparsity-type csr --spmm sputnik --gemm cuBlas --precision half --acc_t fp16 --m $m --k $k --n $n --d 0.25 >> $sputnik_output_file

  # Venom
  ./Samoyeds-Kernel/benchmark/third_party/venom/build/src/benchmark_spmm --sparsity-type n-to-m --spmm spatha --gemm cuBlas --precision half --meta-block-size 32 --block-size 4 --nn_row 2 --mm_row 8 --m $m --k $k --n $n --d 0.5 --bm 128 --bn 64 --bk 32 --wm 32 --wn 64 --wk 32 --mm 16 --mn 8 --mk 32 --nstage 2 --random --check >> $venom_output_file

  # cuSparseLt
  ./Samoyeds-Kernel/benchmark/third_party/venom/build/src/benchmark_spmm  --sparsity-type csr --spmm cuSparseLt_searched --gemm cuBlas --precision half --m $m --k $k --n $n --d 0.5 >> $cusparselt_output_file
done
