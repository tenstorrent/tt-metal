// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward.hpp"

#include "device/softmax_backward_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::normalization {
Tensor ExecuteSoftmaxBackward::invoke(
    const ttnn::Tensor& softmax_output_tensor, const ttnn::Tensor& grad_tensor, int32_t dim) {
    // Normalize dimension to handle negative indices (e.g., -1 for last dimension)
    const auto rank = static_cast<int32_t>(softmax_output_tensor.logical_shape().rank());
    int32_t normalized_dim_signed = dim >= 0 ? dim : rank + dim;

    TT_FATAL(
        normalized_dim_signed >= 0 && normalized_dim_signed < rank,
        "Dimension {} is out of bounds for tensor with rank {}",
        dim,
        rank);

    uint32_t normalized_dim = static_cast<uint32_t>(normalized_dim_signed);

    // Operation
    ttnn::Tensor output_tensor = ttnn::operations::normalization::softmax_backward::softmax_backward(
        softmax_output_tensor, grad_tensor, normalized_dim);

    return output_tensor;
}
}  // namespace ttnn::operations::normalization
