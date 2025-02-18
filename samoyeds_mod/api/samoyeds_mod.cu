/*
 * Copyright (C) 2023 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>

#include "../samoyeds_dir/samoyeds_op.cu"

#include "../sparsifier/sparse_format_converter.hpp"

torch::Tensor ssmm(torch::Tensor A_values,
                   torch::Tensor A_indices,
                   torch::Tensor A_metadata,
                   torch::Tensor B_values,
                   torch::Tensor B_indices,
                   int m,
                   int k,
                   int n,
                   int vector_length,
                   int N,
                   int M) {
//    print_indices((uint *)A_indices.data_ptr(), m, k, vector_length, N, M);
//    print_metadata((uint *)A_metadata.data_ptr(), m, k, vector_length, N, M);

    return ssmm_cuda(A_values, A_indices, A_metadata, B_values, B_indices, m, k, n, vector_length, N, M);
}

torch::Tensor ssmm_trans(torch::Tensor A_values,
                        torch::Tensor A_indices,
                        torch::Tensor A_metadata,
                        torch::Tensor B_values,
                        torch::Tensor B_indices,
                        int m,
                        int k,
                        int n,
                        int vector_length,
                        int N,
                        int M) {
    return ssmm_trans_cuda(A_values, A_indices, A_metadata, B_values, B_indices, m, k, n, vector_length, N, M);
}

torch::Tensor ssmm_fused_silu_trans(torch::Tensor A_values,
                                    torch::Tensor A_indices,
                                    torch::Tensor A_metadata,
                                    torch::Tensor B_values,
                                    torch::Tensor B_indices,
                                    int m,
                                    int k,
                                    int n,
                                    int vector_length,
                                    int N,
                                    int M) {
    return ssmm_fused_silu_trans_cuda(A_values, A_indices, A_metadata, B_values, B_indices, m, k, n, vector_length, N, M);
}

torch::Tensor spmm_dense(torch::Tensor A_values,
                   torch::Tensor A_indices,
                   torch::Tensor A_metadata,
                   torch::Tensor B_values,
                   int m,
                   int k,
                   int n,
                   int vector_length,
                   int N,
                   int M){
    return spmm_dense_cuda(A_values, A_indices, A_metadata, B_values, m, k, n, vector_length, N, M);
}

torch::Tensor spmm_dense_trans( torch::Tensor A_values,
                                torch::Tensor A_indices,
                                torch::Tensor A_metadata,
                                torch::Tensor B_values,
                                int m,
                                int k,
                                int n,
                                int vector_length,
                                int N,
                                int M){
    return spmm_dense_trans_cuda(A_values, A_indices, A_metadata, B_values, m, k, n, vector_length, N, M);
}

torch::Tensor spmm_weighted_dense_trans(torch::Tensor A_values,
                                        torch::Tensor A_indices,
                                        torch::Tensor A_metadata,
                                        torch::Tensor B_values,
                                        torch::Tensor routing_weights,
                                        int m,
                                        int k,
                                        int n,
                                        int vector_length,
                                        int N,
                                        int M){
    return spmm_weighted_dense_trans_cuda(A_values, A_indices, A_metadata, B_values, routing_weights, m, k, n, vector_length, N, M);
}

torch::Tensor spmm_weighted_sparse_trans(torch::Tensor A_values,
                   torch::Tensor A_indices,
                   torch::Tensor A_metadata,
                   torch::Tensor B_values,
                   torch::Tensor B_indices,
                   torch::Tensor routing_weights,
                   int m,
                   int k,
                   int n,
                   int vector_length,
                   int N,
                   int M,
                   int batch_size){
    return spmm_weighted_sparse_trans_cuda(A_values, A_indices, A_metadata, B_values, B_indices, routing_weights, batch_size, m, k, n, vector_length, N, M);
}

void sparsifier(torch::Tensor A_data,
                torch::Tensor A_value,
                torch::Tensor A_indices,
                torch::Tensor A_metadata,
                int A_rows,
                int A_cols,
                int vector_length,
                int N,
                int M) {
    sparsifier_c_impl((half *) A_data.data_ptr(),
                      (half *) A_value.data_ptr(),
                      (uint *) A_indices.data_ptr(),
                      (uint *) A_metadata.data_ptr(),
                      A_rows, A_cols,
                      vector_length, N, M);
}

torch::Tensor get_pruned_value(torch::Tensor A_value,
                               torch::Tensor A_indices,
                               torch::Tensor A_metadata,
                               int A_rows,
                               int A_cols,
                               int vector_length,
                               int N,
                               int M) {

    torch::Tensor A_pruned_value = torch::zeros({A_rows, A_cols}, A_value.dtype());
    pruned_A_c_impl((half *) A_pruned_value.data_ptr(),
                    (half *) A_value.data_ptr(),
                    (uint *) A_indices.data_ptr(),
                    (uint *) A_metadata.data_ptr(),
                    A_rows, A_cols,
                    vector_length, N, M);

    return A_pruned_value;
}

// 如果第一个参数使用TORCH_EXTENSION_NAME，import的模块与pip中的模块名一致
PYBIND11_MODULE(samoyeds_kernel, m) {
    m.def("ssmm",  &ssmm,  "Custom Samoyeds kernel");
    m.def("ssmm_trans",  &ssmm_trans,  "Custom Samoyeds kernel with transpose output");
    m.def("ssmm_fused_silu_trans",  &ssmm_fused_silu_trans,  "Custom Samoyeds kernel with fused silu activation function and transpose output");
    m.def("spmm_dense",  &spmm_dense,  "Custom SPMM kernel, return dense matrix");
    m.def("spmm_dense_trans",  &spmm_dense_trans,  "Custom SPMM kernel, return transpose dense matrix");
    m.def("spmm_weighted_dense_trans",  &spmm_weighted_dense_trans,  "Custom SPMM kernel, return weighted transpose dense matrix");
    m.def("spmm_weighted_sparse_trans",  &spmm_weighted_sparse_trans,  "Custom SPMM kernel, return weighted transpose sparse matrix");
    m.def("sparsifier",  &sparsifier,  "Custom Samoyeds Sparsifier");
    m.def("get_pruned_value",  &get_pruned_value,  "Custom Samoyeds Pruned Value Getter");
}