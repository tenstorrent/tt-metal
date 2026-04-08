
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {

Tensor moreh_nll_loss(
    const Tensor& input_tensor,
    const Tensor& target_tensor,
    const std::string& reduction,
    const std::optional<Tensor>& weight_tensor = std::nullopt,
    const std::optional<Tensor>& divisor_tensor = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    int32_t ignore_index = -100,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
