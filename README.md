# Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores (EuroSys'25)
``Samoyeds`` is an innovative acceleration system for MoE LLMs utilizing Sparse Tensor Cores (SpTCs).

## Install

### Pre-requisites
Samoyeds requires the following dependencies:
- CUDA 11.4+
- CMake 3.18+
- GPUs with Sparse Tensor Core (such as NVIDIA GPUs with Ampere architecture or newer).

### Build & Install

```shell
docker pull kevinwu2017/samoyeds:1.0.0
docker run -it --gpus all --name samoyeds-ae kevinwu2017/samoyeds:1.0.0
```

## Reproduction

The hardware requirements for each experiment are as follows:
- Experiments (1), (2), (3), and (6): These experiments can be conducted on a single GPU, such as the NVIDIA GeForce RTX 4070 Super used in our paper.
- Experiment (4): This experiment involves post-training of models, which may require high-end GPUs such as the A100-80G used in our paper.
- Experiment (5): This experiment analyzes performance portability and requires multiple GPUs with different architectures (e.g., RTX 3090, RTX 4070 Super, RTX 4090, and A100, as used in our paper).


#### (1) To reproduce the kernel level results (Figure 12, 13)

```shell
./artifacts/kernel/synthetic_scripts.sh
./artifacts/kernel/kernel_model_config_scripts.sh
```

The plotting scripts are located at:

- ``./artifacts/kernel/figure12_plot.ipynb``

- ``./artifacts/kernel/figure13_plot.ipynb``

After executing these scripts, you can generate the corresponding figures using our provided code.

#### (2) To reproduce the MoE module level results (Figure 14)
```shell
./artifacts/MoE/figure14_scripts.sh
```

The plotting script is located at:

- ``./artifacts/MoE/figure14_plot.ipynb``

After executing the script, you can generate the corresponding figure using our provided code.

#### (3) To reproduce the end-to-end level results (Figure 15, 16)
```shell
./artifacts/model/figure15_scripts.sh
./artifacts/model/figure16_scripts.sh
```
The plotting scripts are located at:

- ``./artifacts/model/figure15_plot.ipynb``

- ``./artifacts/model/figure16_plot.ipynb``

After executing these scripts, you can generate the corresponding figures using our provided code.

#### (4) To reproduce the breakdown analysis results (Figure 17)

```shell
./artifacts/MoE/figure17_scripts.sh
```

The plotting script is located at:

- ``./artifacts/MoE/figure17_plot.ipynb``

After executing the script, you can generate the corresponding figure using our provided code.

#### (5) To reproduce the results of model accuracy (Table 4, 5)

The scripts includes experiments with configurations using the pair-wise version of the sparsifier.

> The following scripts require execution on high-memory GPUs or multi-GPU configurations. Specifically:
> - The script for collecting data in Table 4 is configured to utilize a cluster of 4 GPUs.
> - The scripts for collecting data in Table 5 must be run on an NVIDIA A100 80GB GPU to avoid Out-Of-Memory (OOM) errors. Lower-capacity GPUs may not have sufficient memory to handle these operations.

```shell
cd sparseml
# Table 4
bach benchmark/scripts/samoyeds_gradual_pair.sh
# Table 5
bash benchmark/scripts/samoyeds_qwen2_80G.sh
bash benchmark/scripts/samoyeds_tiny_llama_80G.sh
```
The results are stored in the ``benchmark/output_dir/`` directory.

#### (6) To reproduce the performance portability results of Samoyeds (Figure 18)

> To plot figure 18, the following script need to run on multiple GPUs, including NVIDIA GeForce RTX 3070, NVIDIA GeForce RTX 4070 Super, NVIDIA GeForce RTX 4090, and NVIDIA A100. 

```shell
./artifacts/kernel/synthetic_scripts.sh
```

> Figure 18 can be reproduced by collecting results on different GPUs into ``./artifacts/results/kernel/`` folder.

The plotting script is located at:

- ``./artifacts/kernel/figure18_plot.ipynb``

After executing the script, you can generate the corresponding figure using our provided code.
