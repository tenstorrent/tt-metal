// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

Tensor moreh_norm(
    const Tensor& input,
    float p,
    std::optional<std::variant<int64_t, ttnn::SmallVector<int64_t>>> dim = std::nullopt,
    bool keepdim = false,
    const std::optional<Tensor>& output = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
