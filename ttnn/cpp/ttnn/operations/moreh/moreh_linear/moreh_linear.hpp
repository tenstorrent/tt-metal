// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::operations::moreh::moreh_linear {
struct MorehLinear {
    static Tensor invoke(
        const Tensor& input,
        const Tensor& weight,
        const std::optional<Tensor>& bias,
        const std::optional<Tensor>& output,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig> compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_linear

namespace ttnn {
constexpr auto moreh_linear = ttnn::
    register_operation_with_auto_launch_op<"ttnn::moreh_linear", ttnn::operations::moreh::moreh_linear::MorehLinear>();
}
