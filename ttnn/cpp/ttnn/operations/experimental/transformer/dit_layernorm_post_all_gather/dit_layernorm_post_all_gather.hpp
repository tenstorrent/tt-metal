// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental {

// Consumes gathered Welford stats and applies LayerNorm with optional gamma/beta.
ttnn::Tensor dit_layernorm_post_allgather(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& stats,
    float epsilon = 1e-5,
    const std::optional<const ttnn::Tensor>& weight = std::nullopt,
    const std::optional<const ttnn::Tensor>& bias = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<const DataType>& dtype = std::nullopt);

}  // namespace ttnn::experimental
