// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn {

std::vector<std::optional<Tensor>> moreh_bmm_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mat2,
    const std::vector<bool>& are_required_outputs = {true, true},
    const std::optional<Tensor>& input_grad = std::nullopt,
    const std::optional<Tensor>& mat2_grad = std::nullopt,
    const std::optional<MemoryConfig>& input_grad_memory_config = std::nullopt,
    const std::optional<MemoryConfig>& mat2_grad_memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
