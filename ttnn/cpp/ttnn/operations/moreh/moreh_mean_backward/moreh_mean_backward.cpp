// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_mean_backward.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/moreh/moreh_mean_backward/device/moreh_mean_backward_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_mean_backward {
Tensor MorehMeanBackward::invoke(
    const Tensor& output_grad,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keepdim,
    const std::optional<Shape>& input_grad_shape,
    const std::optional<Tensor>& input_grad,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    auto output_grad_rank = output_grad.get_shape().rank();
    auto input_grad_rank = output_grad_rank;
    if (keepdim == false) {
        if (!dim.has_value()) {
            // do nothing
        } else if (std::holds_alternative<int64_t>(dim.value())) {
            input_grad_rank += 1;
        } else {
            auto dims = std::get<std::vector<int64_t>>(dim.value());
            input_grad_rank += dims.size();
        }
    }
    std::vector<int64_t> dims = tt::operations::primary::get_dim(dim, input_grad_rank);
    return ttnn::prim::moreh_mean_backward(
        output_grad, dims, keepdim, input_grad_shape, input_grad, memory_config, compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_mean_backward
