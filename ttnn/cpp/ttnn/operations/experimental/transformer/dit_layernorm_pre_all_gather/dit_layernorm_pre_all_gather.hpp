// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental {

// Computes Welford stats (sum and sumsq) over the last dim for LayerNorm.
ttnn::Tensor dit_layernorm_pre_allgather(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& recip_tensor,
    DataType dtype = DataType::BFLOAT16,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
