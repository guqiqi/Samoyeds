/*
 * Copyright (c) 2025 Chenpeng Wu (cpwu_sjtu@sjtu.edu.cn), Qiqi Gu (qiqi.gu@sjtu.edu.cn). 
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>

#include "mm_utils/spmm_utils.h"
#include "ssmm/horizontal_spmm_kernel_op.h"
#include "ssmm/horizontal_ssmm_kernel_op.h"

torch::Tensor ssmm_cuda(torch::Tensor A_values,
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
    using BlockShape = ShapeBase<128, 32, 64>;
    using WarpShape = ShapeBase<32, 32, 64>;
    using MmaShape = ShapeBase<16, 32, 8>;

    // A_values = A_values.contiguous();
    // A_indices = A_indices.contiguous();
    // A_metadata = A_metadata.contiguous();
    // B_values = B_values.contiguous();
    // B_indices = B_indices.contiguous();

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(A_values.device()).requires_grad(false);
    torch::Tensor C = torch::zeros({m, B_indices.size(0)}, options);

    HorizontalSsmmKernelExec<BlockShape, WarpShape, MmaShape, 2>(
            m, n, k, vector_length, N, M,
            (half*)A_values.data_ptr(),
            (uint*)A_metadata.data_ptr(),
            (uint*)A_indices.data_ptr(),
            (half*)B_values.data_ptr(),
            (uint*)B_indices.data_ptr(),
            static_cast<uint>(B_indices.size(0)),
            (half*)C.data_ptr(),
            (half*)C.data_ptr());

    return C;
}

torch::Tensor ssmm_trans_cuda(  torch::Tensor A_values,
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
    using BlockShape = ShapeBase<128, 32, 64>;
    using WarpShape = ShapeBase<32, 32, 64>;
    using MmaShape = ShapeBase<16, 32, 8>;

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(A_values.device()).requires_grad(false);
    torch::Tensor C = torch::zeros({B_indices.size(0), m}, options);

    HorizontalSsmmTransKernelExec<BlockShape, WarpShape, MmaShape, 2>(
            m, n, k, vector_length, N, M,
            (half*)A_values.data_ptr(),
            (uint*)A_metadata.data_ptr(),
            (uint*)A_indices.data_ptr(),
            (half*)B_values.data_ptr(),
            (uint*)B_indices.data_ptr(),
            static_cast<uint>(B_indices.size(0)),
            (half*)C.data_ptr(),
            (half*)C.data_ptr());

    return C;
}

torch::Tensor ssmm_fused_silu_trans_cuda(torch::Tensor A_values,
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
    using BlockShape = ShapeBase<128, 32, 64>;
    using WarpShape = ShapeBase<32, 32, 64>;
    using MmaShape = ShapeBase<16, 32, 8>;

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(A_values.device()).requires_grad(false);
    torch::Tensor C = torch::zeros({B_indices.size(0), m}, options);

    HorizontalSsmmFusedActTransKernelExec<BlockShape, WarpShape, MmaShape, 2>(
            m, n, k, vector_length, N, M,
            (half*)A_values.data_ptr(),
            (uint*)A_metadata.data_ptr(),
            (uint*)A_indices.data_ptr(),
            (half*)B_values.data_ptr(),
            (uint*)B_indices.data_ptr(),
            static_cast<uint>(B_indices.size(0)),
            (half*)C.data_ptr(),
            (half*)C.data_ptr());

    return C;
}

torch::Tensor spmm_dense_cuda(  torch::Tensor A_values,
                                torch::Tensor A_indices,
                                torch::Tensor A_metadata,
                                torch::Tensor B_values,
                                int m,
                                int k,
                                int n,
                                int vector_length,
                                int N,
                                int M) {
    using BlockShape = ShapeBase<128, 32, 64>;
    using WarpShape = ShapeBase<32, 32, 64>;
    using MmaShape = ShapeBase<16, 32, 8>;

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(A_values.device()).requires_grad(false);
    torch::Tensor D = torch::zeros({m, n}, options);

    HorizontalSpmmKernelExec<BlockShape, WarpShape, MmaShape, 2>(
            m, n, k, vector_length, N, M,
            (half*)A_values.data_ptr(),
            (uint*)A_metadata.data_ptr(),
            (uint*)A_indices.data_ptr(),
            (half*)B_values.data_ptr(),
            (half*)D.data_ptr(),
            (half*)D.data_ptr());

    return D;
}

torch::Tensor spmm_dense_trans_cuda(    torch::Tensor A_values,
                                        torch::Tensor A_indices,
                                        torch::Tensor A_metadata,
                                        torch::Tensor B_values,
                                        int m,
                                        int k,
                                        int n,
                                        int vector_length,
                                        int N,
                                        int M) {
    using BlockShape = ShapeBase<128, 32, 64>;
    using WarpShape = ShapeBase<32, 32, 64>;
    using MmaShape = ShapeBase<16, 32, 8>;

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(A_values.device()).requires_grad(false);
    torch::Tensor D = torch::zeros({n, m}, options);

    HorizontalSpmmTransKernelExec<BlockShape, WarpShape, MmaShape, 2>(
            m, n, k, vector_length, N, M,
            (half*)A_values.data_ptr(),
            (uint*)A_metadata.data_ptr(),
            (uint*)A_indices.data_ptr(),
            (half*)B_values.data_ptr(),
            (half*)D.data_ptr(),
            (half*)D.data_ptr());

    return D;
}

torch::Tensor spmm_weighted_dense_trans_cuda(   torch::Tensor A_values,
                                                torch::Tensor A_indices,
                                                torch::Tensor A_metadata,
                                                torch::Tensor B_values,
                                                torch::Tensor routing_weights,
                                                int m,
                                                int k,
                                                int n,
                                                int vector_length,
                                                int N,
                                                int M) {
    using BlockShape = ShapeBase<128, 32, 64>;
    using WarpShape = ShapeBase<32, 32, 64>;
    using MmaShape = ShapeBase<16, 32, 8>;

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(A_values.device()).requires_grad(false);
    torch::Tensor D = torch::zeros({n, m}, options);

    HorizontalSpmmDenseWeightedTransKernelExec<BlockShape, WarpShape, MmaShape, 2>(
            m, n, k, vector_length, N, M,
            (half*)A_values.data_ptr(),
            (uint*)A_metadata.data_ptr(),
            (uint*)A_indices.data_ptr(),
            (half*)B_values.data_ptr(),
            (half*)routing_weights.data_ptr(),
            (half*)D.data_ptr(),
            (half*)D.data_ptr());

    return D;
}

torch::Tensor spmm_weighted_sparse_trans_cuda(  torch::Tensor A_values,
                                                torch::Tensor A_indices,
                                                torch::Tensor A_metadata,
                                                torch::Tensor B_values,
                                                torch::Tensor B_indices,
                                                torch::Tensor routing_weights,
                                                int batch_size,
                                                int m,
                                                int k,
                                                int n,
                                                int vector_length,
                                                int N,
                                                int M) {
    using BlockShape = ShapeBase<128, 32, 64>;
    using WarpShape = ShapeBase<32, 32, 64>;
    using MmaShape = ShapeBase<16, 32, 8>;

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(A_values.device()).requires_grad(false);
    torch::Tensor D = torch::zeros({batch_size, m}, options);

    HorizontalSpmmSparseWeightedTransKernelExec<BlockShape, WarpShape, MmaShape, 2>(
            m, n, k, vector_length, N, M,
            (half*)A_values.data_ptr(),
            (uint*)A_metadata.data_ptr(),
            (uint*)A_indices.data_ptr(),
            (half*)B_values.data_ptr(),
            (uint*)B_indices.data_ptr(),
            static_cast<uint>(B_indices.size(0)),
            (half*)routing_weights.data_ptr(),
            (half*)D.data_ptr(),
            (half*)D.data_ptr());

    return D;
}