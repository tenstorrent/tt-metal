/*
 * SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

Tensor moreh_nll_loss_unreduced(
    const Tensor &input_tensor,
    const Tensor &target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> output_tensor,
    const int32_t ignore_index,
    const MemoryConfig &memory_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary
}  // namespace operations
}  // namespace tt
