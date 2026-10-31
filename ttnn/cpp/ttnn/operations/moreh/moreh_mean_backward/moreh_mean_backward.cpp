// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_mean_backward.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/moreh/moreh_mean_backward/device/moreh_mean_backward_device_operation.hpp"

namespace ttnn {

Tensor moreh_mean_backward(
    const Tensor& output_grad,
    std::optional<std::variant<int64_t, ttsl::SmallVector<int64_t>>> dim,
    const bool keepdim,
    const std::optional<ttnn::Shape>& input_grad_shape,
    const std::optional<Tensor>& input_grad,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    auto input_grad_rank = output_grad.logical_shape().rank();
    if (input_grad.has_value()) {
        input_grad_rank = input_grad->logical_shape().rank();
    } else if (input_grad_shape.has_value()) {
        input_grad_rank = input_grad_shape->rank();
    }
    ttsl::SmallVector<int64_t> dims = ttnn::operations::get_dim(dim, input_grad_rank);
    return ttnn::prim::moreh_mean_backward(
        output_grad, dims, keepdim, input_grad_shape, input_grad, memory_config, compute_kernel_config);
}

}  // namespace ttnn
