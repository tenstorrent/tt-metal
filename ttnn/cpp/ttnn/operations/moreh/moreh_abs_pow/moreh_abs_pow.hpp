// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_abs_pow {
struct MorehAbsPow {
    static Tensor invoke(
        const Tensor& input,
        float p,
        const std::optional<Tensor>& output,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_abs_pow

namespace ttnn {
constexpr auto moreh_abs_pow =
    ttnn::register_operation<"ttnn::moreh_abs_pow", ttnn::operations::moreh::moreh_abs_pow::MorehAbsPow>();
}
