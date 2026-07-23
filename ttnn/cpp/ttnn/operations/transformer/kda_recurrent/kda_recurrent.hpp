// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::transformer {

/**
 * One fused recurrent Kimi Delta Attention state step.
 *
 * All tensors must be FLOAT32, TILE_LAYOUT, and interleaved DRAM.
 * q_scaled and k_unit are normalized before this private recurrence boundary;
 * decay is exp(log_gate), stored as a column so it broadcasts over state rows.
 *
 * Args:
 *   q_scaled [BH, 1, K]
 *   k_unit   [BH, 1, K]
 *   v        [BH, 1, V]
 *   decay    [BH, K, 1]
 *   beta     [BH, 1, 1]
 *   state    [BH, K, V]
 *
 * Returns:
 *   (output [BH, 1, V], final_state [BH, K, V])
 */
std::tuple<ttnn::Tensor, ttnn::Tensor> kda_recurrent_step(
    const ttnn::Tensor& q_scaled,
    const ttnn::Tensor& k_unit,
    const ttnn::Tensor& v,
    const ttnn::Tensor& decay,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& state,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
