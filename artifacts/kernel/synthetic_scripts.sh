#!/bin/bash

gpu_id=${CUDA_VISIBLE_DEVICES:-0}
gpu_model=$(nvidia-smi -i $gpu_id --query-gpu=gpu_name --format=csv,noheader,nounits)
gpu_model=${gpu_model// /.}
cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)

echo "GPU model: $gpu_model"
echo "CUDA version: $cuda_version"

m_values=(256 512 1024 2048 4096 8192 16384)
k_values=(256 512 1024 2048 4096 8192 16384)
n_values=(256 512 1024 2048 4096 8192 16384)

output_dir="artifacts/results/kernel"
mkdir -p $output_dir

# SSMM
echo "benchmarking ssmm"
output_file="$output_dir/SSMM_${gpu_model}.txt"
: >| $output_file

for m in "${m_values[@]}"; do
  for n in "${n_values[@]}"; do
    for k in "${k_values[@]}"; do
      echo "Running with m=$m, n=$n, k=$k" >> $output_file
      ./Samoyeds-Kernel/build/benchmark/benchmark -m $m -n $n -k $k -N 1 -M 2 --vector_length 128 --method SSMM >> $output_file
    done
  done
done

# Sputnik
echo "benchmarking Sputnik and cuBlas"
output_file="$output_dir/Sputnik_and_cuBlas_${gpu_model}.txt"
: >| $output_file

echo "algo,arch,m,k,n,meta_block_sz,block_sz,nn_row,mm_col,density,bm,bn,bk,wm,wn,wk,mm,mn,mk,nstage,spmm_time,gemm_time,speedup,error" > $output_file

for m in "${m_values[@]}"; do
  for n in "${n_values[@]}"; do
    for k in "${k_values[@]}"; do
      ./Samoyeds-Kernel/benchmark/third_party/venom/build/src/benchmark_spmm --sparsity-type csr --spmm sputnik --gemm cuBlas --precision half --acc_t fp16 --m $m --k $k --n $n --d 0.25 >> $output_file
    done
  done
done

# Venom
echo "benchmarking venom and cuBlas"
output_file="$output_dir/Venom_and_cuBlas_${gpu_model}.txt"
: >| $output_file

echo "algo,arch,m,k,n,meta_block_sz,block_sz,nn_row,mm_col,density,bm,bn,bk,wm,wn,wk,mm,mn,mk,nstage,spmm_time,gemm_time,speedup,error" > $output_file

for m in "${m_values[@]}"; do
  for n in "${n_values[@]}"; do
    for k in "${k_values[@]}"; do
      ./Samoyeds-Kernel/benchmark/third_party/venom/build/src/benchmark_spmm --sparsity-type n-to-m --spmm spatha --gemm cuBlas --precision half --meta-block-size 32 --block-size 4 --nn_row 2 --mm_row 8 --m $m --k $k --n $n --d 0.5 --bm 128 --bn 64 --bk 32 --wm 32 --wn 64 --wk 32 --mm 16 --mn 8 --mk 32 --nstage 2 --random --check >> $output_file
    done
  done
done

# cuSparseLtsearched
echo "benchmarking cuSparseLt searched and cuBlas"
output_file="$output_dir/cuSparseLtsearched_and_cuBlas_${gpu_model}.txt"
: >| $output_file

echo "algo,arch,m,k,n,meta_block_sz,block_sz,nn_row,mm_col,density,bm,bn,bk,wm,wn,wk,mm,mn,mk,nstage,spmm_time,gemm_time,speedup,error" > $output_file

for m in "${m_values[@]}"; do
  for n in "${n_values[@]}"; do
    for k in "${k_values[@]}"; do
      ./Samoyeds-Kernel/benchmark/third_party/venom/build/src/benchmark_spmm --sparsity-type csr --spmm cuSparseLt_searched --gemm cuBlas --precision half --m $m --k $k --n $n --d 0.5 >> $output_file
    done
  done
done
