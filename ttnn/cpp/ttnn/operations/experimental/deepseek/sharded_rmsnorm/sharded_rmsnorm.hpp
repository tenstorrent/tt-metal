// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek {

// Fast single-row RMSNorm for a [1, 1, 1, D] BFLOAT16 WIDTH_SHARDED L1 tensor.
// Computes y = x * rsqrt(mean(x^2) + epsilon) * weight across all N cores of input_tensor's
// shard grid, optionally outputting ROW_MAJOR or TILE in L1 WIDTH_SHARDED or INTERLEAVED memory.
Tensor sharded_rmsnorm(
    const Tensor& input_tensor,
    const Tensor& weight,
    float epsilon = 1e-6f,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Layout> output_layout = std::nullopt);

}  // namespace ttnn::experimental::deepseek
