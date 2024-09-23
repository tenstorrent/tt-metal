// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_norm {
struct MorehNorm {
    static Tensor invoke(
        const Tensor& input,
        float p,
        std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
        bool keepdim,
        const std::optional<Tensor>& output,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_norm

namespace ttnn {
constexpr auto moreh_norm =
    ttnn::register_operation_with_auto_launch_op<"ttnn::moreh_norm", ttnn::operations::moreh::moreh_norm::MorehNorm>();
}
