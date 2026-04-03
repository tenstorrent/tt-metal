// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

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
