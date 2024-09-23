
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
namespace ttnn::operations::moreh::moreh_dot {

struct MorehDot {
    static Tensor invoke(
        const Tensor& input_a,
        const Tensor& input_b,
        const std::optional<Tensor>& output,
        const std::optional<DataType>& dtype,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_dot

namespace ttnn {
constexpr auto moreh_dot =
    ttnn::register_operation_with_auto_launch_op<"ttnn::moreh_dot", ttnn::operations::moreh::moreh_dot::MorehDot>();
}
