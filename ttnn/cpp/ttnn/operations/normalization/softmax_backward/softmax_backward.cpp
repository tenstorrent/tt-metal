// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward.hpp"

#include "device/softmax_backward_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::normalization {
Tensor ExecuteSoftmaxBackward::invoke(
    const ttnn::Tensor& softmax_output_tensor, const ttnn::Tensor& grad_tensor, uint32_t dim) {
    // Operation
    ttnn::Tensor output_tensor =
        ttnn::operations::normalization::softmax_backward::softmax_backward(softmax_output_tensor, grad_tensor, dim);

    return output_tensor;
}
}  // namespace ttnn::operations::normalization
