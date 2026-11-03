// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

[[deprecated("moreh_bmm is deprecated and will be removed after 2026-11-26. Use ttnn::matmul instead.")]]
Tensor moreh_bmm(
    const Tensor& input,
    const Tensor& mat2,
    const std::optional<Tensor>& output = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
