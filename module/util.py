import torch
import torch.nn.functional as F

import samoyeds_kernel

M = 2
N = 1
vector_length = 128
SPTC_M = 4
SPTC_N = 2
NUM_OF_META_PER_UINT = 16

def sparsifier(tensor):
    nrows, ncols = tensor.shape
    A_values = torch.zeros((nrows // M * N, ncols // SPTC_M * SPTC_N), dtype=tensor.dtype, device='cpu')
    A_indices = torch.zeros((ncols // vector_length, nrows // M * N), dtype=torch.int32, device='cpu')
    A_metadata = torch.zeros((nrows // M * N, ncols // SPTC_M * SPTC_N // NUM_OF_META_PER_UINT), dtype=torch.int32,
                             device='cpu')
    tensor_clone = tensor.cpu().detach().clone()
    samoyeds_kernel.sparsifier(tensor_clone,
                        A_values,
                        A_indices,
                        A_metadata,
                        nrows, ncols, vector_length, N, M)
    A_values = A_values.to(tensor.device)
    A_indices = A_indices.to(tensor.device)
    A_metadata = A_metadata.to(tensor.device)
    return A_values, A_indices, A_metadata

def padding_idx(tensor, align):
    target_length = ((tensor.size(0) - 1) // align + 1) * align  # 计算目标长度为 align 的倍数
    padded_tensor = F.pad(tensor, (0, target_length - tensor.size(0)), value=0).to(torch.int32)  # 进行填充
    return padded_tensor

def padding_weight(tensor, align):
    target_length = ((tensor.size(0) - 1) // align + 1) * align  # 计算目标长度为 align 的倍数
    padded_tensor = F.pad(tensor, (0, 0, 0, target_length - tensor.size(0)), value=0)  # 进行填充
    return padded_tensor

def padding_input(tensor, align):
    # target_length = ((tensor.size(0) - 1) // align + 1) * align  # 计算目标长度为 align 的倍数
    padding_length = (align - (tensor.size(0) % align)) % align
    padded_tensor = F.pad(tensor, (0, 0, 0, padding_length))  # 进行填充
    return padded_tensor