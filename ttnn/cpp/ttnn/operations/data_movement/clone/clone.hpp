// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::data_movement::clone {
struct Clone {
    static Tensor invoke(
        const Tensor& input,
        const std::optional<DataType>& dtype,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::data_movement::clone

namespace ttnn {
constexpr auto clone = ttnn::register_operation<"ttnn::clone", ttnn::operations::data_movement::clone::Clone>();
}  // namespace ttnn
