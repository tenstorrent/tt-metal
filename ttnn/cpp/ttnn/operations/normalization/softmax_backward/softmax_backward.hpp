// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #include "device/softmax_operation_types.hpp"

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include <optional>

namespace ttnn::operations::normalization {
/**
 * @brief Executes the backpropagation on softmax operation on a tensor along a specified dimension.
 *
 * Computes softmax_backward(y, grad, dim) = y * (grad - (y * grad).sum(dim, keepdim=True)) along the specified
 * dimension. The operation creates a new output tensor.
 */
struct ExecuteSoftmaxBackward {
    static ttnn::Tensor invoke(const ttnn::Tensor& softmax_output_tensor, const ttnn::Tensor& grad_tensor, int32_t dim);
};
}  // namespace ttnn::operations::normalization

namespace ttnn {

constexpr auto softmax_backward =
    ttnn::register_operation<"ttnn::softmax_backward", ttnn::operations::normalization::ExecuteSoftmaxBackward>();

}  // namespace ttnn
