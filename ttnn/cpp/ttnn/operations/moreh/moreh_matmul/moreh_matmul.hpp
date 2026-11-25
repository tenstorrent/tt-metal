// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

[[deprecated("moreh_matmul is deprecated and will be removed after 2026-12-16. Use ttnn::matmul instead.")]]
Tensor moreh_matmul(
    const Tensor& input,
    const Tensor& other,
    bool transpose_input = false,
    bool transpose_other = false,
    const std::optional<Tensor>& output = std::nullopt,
    const std::optional<const Tensor>& bias = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
