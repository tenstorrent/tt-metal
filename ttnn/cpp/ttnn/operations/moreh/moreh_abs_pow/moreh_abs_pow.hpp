// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::operations::moreh::moreh_abs_pow {
struct MorehAbsPow {
    static Tensor invoke(
        const Tensor& input,
        const float p,
        const std::optional<Tensor>& output,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_abs_pow

namespace ttnn {
constexpr auto moreh_abs_pow = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_abs_pow",
    ttnn::operations::moreh::moreh_abs_pow::MorehAbsPow>();
}
