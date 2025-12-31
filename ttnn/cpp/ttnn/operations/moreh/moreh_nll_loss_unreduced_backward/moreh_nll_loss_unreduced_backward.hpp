// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using tt::tt_metal::MemoryConfig;

namespace ttnn {

Tensor moreh_nll_loss_unreduced_backward(
    const Tensor& target_tensor,
    const Tensor& output_grad_tensor,
    const std::optional<Tensor>& weight_tensor = std::nullopt,
    const std::optional<Tensor>& input_grad_tensor = std::nullopt,
    int32_t ignore_index = -100,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn
