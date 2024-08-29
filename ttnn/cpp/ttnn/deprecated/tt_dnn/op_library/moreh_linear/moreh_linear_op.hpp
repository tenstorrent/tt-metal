/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

Tensor moreh_linear(
    const Tensor& input,
    const Tensor& weight,
    std::optional<const Tensor> bias = std::nullopt,
    std::optional<const Tensor> output = std::nullopt,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);


}  // namespace primary

}  // namespace operations

}  // namespace tt
