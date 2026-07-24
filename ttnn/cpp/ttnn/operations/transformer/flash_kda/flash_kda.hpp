// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::transformer {

/**
 * Flash KDA (Kimi Delta Attention) recurrent state update — single step, one core per item.
 *
 * Computes, per item:
 *   S_tilde = S_prev * g            (g varies per key-dim row, replicated across value columns)
 *   pred    = k @ S_tilde
 *   err     = v - pred
 *   delta   = beta * err
 *   S_new   = S_tilde + (k outer delta)
 *   out     = q @ S_new
 *
 * All inputs must be float32, TILE_LAYOUT, DRAM.
 *
 * Args:
 *   S_prev [N, Dk, Dv]   previous recurrent state
 *   g      [N, Dk, 1]    per-key-dim-row decay (column layout)
 *   k      [N, 1, Dk]    key vector (row layout)
 *   v      [N, 1, Dv]    value vector (row layout)
 *   beta   [N, 1, 1]     per-item scalar gate
 *   q      [N, 1, Dk]    query vector (row layout)
 *
 * Returns:
 *   (S_new [N, Dk, Dv], out [N, 1, Dv])
 */
std::tuple<ttnn::Tensor, ttnn::Tensor> flash_kda(
    const ttnn::Tensor& S_prev,
    const ttnn::Tensor& g,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& q,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
