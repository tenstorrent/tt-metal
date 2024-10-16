// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
namespace ttnn::operations::moreh::moreh_mean_backward {
struct MorehMeanBackward {
    static Tensor invoke(const Tensor& output_grad,
                         std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
                         const bool keepdim,
                         const std::optional<Shape>& input_grad_shape,
                         const std::optional<Tensor>& input_grad,
                         const std::optional<MemoryConfig>& memory_config,
                         const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_mean_backward

namespace ttnn {
constexpr auto moreh_mean_backward =
    ttnn::register_operation_with_auto_launch_op<"ttnn::moreh_mean_backward",
                                                 ttnn::operations::moreh::moreh_mean_backward::MorehMeanBackward>();
}
