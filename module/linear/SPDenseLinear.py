import torch.nn as nn
import torch

from module.util import sparsifier, N, M, vector_length, padding_input
import samoyeds_kernel

def sparse_dense_mul_dense_dispatch(A_sparse_values, A_sparse_indices, A_sparse_metadata,
                                                B_values, m, k, n):
    output = samoyeds_kernel.spmm_dense( A_sparse_values,  # lhs_matrix, compressed values
                                   A_sparse_indices,  # indices
                                   A_sparse_metadata,  # metadata
                                   B_values,  # rhs_matrix, sparse inputs
                                   m,  # m
                                   k,  # k
                                   n,  # n
                                   vector_length,  # vector_length
                                   N,  # structure_N
                                   M,  # structure_M
                                   )

    return output

class SPDenseLinear(nn.Module):
    def __init__(self, original = None, in_features = 1024, out_features = 4096):
        super().__init__()
        if original is None:
            self.in_features = in_features
            self.out_features = out_features
            self.weight, self.indices, self.metadata = sparsifier(nn.Parameter(torch.rand(out_features, in_features)))
        else:
            self.bias = original.bias
            self.in_features = original.in_features
            self.out_features = original.out_features
            self.weight, self.indices, self.metadata = sparsifier(original.weight)

        self.weight = nn.Parameter(self.weight)
        self.indices = nn.Parameter(self.indices, requires_grad=False)
        self.metadata = nn.Parameter(self.metadata, requires_grad=False)

    def forward(self, input):
        input = padding_input(input, 64)
        return sparse_dense_mul_dense_dispatch(self.weight, self.indices, self.metadata, input.T,
                                          self.weight.shape[0] * 2,
                                          self.weight.shape[1] * 2, input.shape[0])

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )