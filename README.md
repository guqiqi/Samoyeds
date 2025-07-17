# Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores (EuroSys'25)

This repository contains the implementation of **Samoyeds**, an innovative acceleration system for MoE LLMs utilizing Sparse Tensor Cores (SpTCs). Our work has been published at EuroSys'25.

Samoyeds is the first to apply sparsity simultaneously to both activations and model parameters. It introduces a bespoke sparse data format tailored for MoE computation and develops a specialized sparse-sparse matrix multiplication kernel. Furthermore, Samoyeds incorporates systematic optimizations specifically designed for the execution of dual-side structured sparse MoE LLMs on SpTCs, further enhancing system performance.

**Paper**: [Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores](https://dl.acm.org/doi/10.1145/3689031.3717455)

## Install

### Pre-requisites
Samoyeds requires the following dependencies:
- CUDA 11.4+
- CMake 3.18+
- GPUs with Sparse Tensor Core (such as NVIDIA GPUs with Ampere architecture or newer).

### Build & Install

#### Option 1: pull a pre-built docker image from dockerhub
```shell
docker pull kevinwu2017/samoyeds:1.0.0
docker run -it --gpus all --name samoyeds-ae kevinwu2017/samoyeds:1.0.0
```

#### Option 2: build Samoyeds from source code
```shell
git clone --recurse-submodules https://github.com/guqiqi/Samoyeds.git
cd Samoyeds

conda create --name samoyeds python=3.10
conda activate samoyeds

./build.sh
```

### Run

#### Dual-Sparse Kernel

Run SSMM kernel with the Mixtral model config:
```shell
./Samoyeds-Kernel/build/benchmark/benchmark -m 14336 -n 4096 -k 4096 -N 1 -M 2 --vector_length 128 --method SSMM
```

#### MoE Module

Run Samoyeds MoE module with Mixtral model config:
```shell
python mixtral_Samoyeds.py --time --batch_size 1 --mlp --experts 8 --hidden_size 4096 --intermediate_size 14336 --seq_len 4096
```

#### End-to-End

Run Samoyeds with Mixtral model config:
```shell
python mixtral_Samoyeds.py --time --batch_size 1 --layer --flash --experts 8 --hidden_size 4096 --intermediate_size 14336
```

## LICENCE

This project is licensed under the Apache License 2.0. See the [LICENCE](./LICENCE) file for details.

## Citation

If you use Samoyeds in your research, please cite our paper:

```bibtex
@inproceedings{2025samoyeds,
  title={Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores},
  author={Wu, Chenpeng and Gu, Qiqi and Shi, Heng and Yao, Jianguo and Guan, Haibing},
  booktitle={Proceedings of the Twentieth European Conference on Computer Systems},
  pages={293--310},
  year={2025}
}
```

## Contact

For questions or collaboration, please feel free to contact:
- [cpwu_sjtu@sjtu.edu.cn](mailto:cpwu_sjtu@sjtu.edu.cn)
- [qiqi.gu@sjtu.edu.cn](mailto:qiqi.gu@sjtu.edu.cn)
