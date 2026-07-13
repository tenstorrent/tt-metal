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
 * Gated DeltaNet attention — sequential inter-chunk scan (Path A).
 *
 * Python pre-normalises L_unit to unit-diagonal form and precomputes L_inv
 * (4 diagonal block inverses per chunk). The C++ kernel performs blocked
 * forward substitution and the sequential inter-chunk state update.
 *
 * All inputs must be float32, TILE_LAYOUT, DRAM.
 *
 * Args:
 *   L_unit       [BH, NC, C, C]    unit-diagonal lower-tri (= D^{-1}*L_mat)
 *   v_beta_sc    [BH, NC, C, Dv]   D^{-1} @ v_beta
 *   k_bd_sc      [BH, NC, C, Dk]   D^{-1} @ k_beta_decay
 *   intra_attn   [BH, NC, C, C]    intra-chunk attention (q@k.T * mask)
 *   q_decay      [BH, NC, C, Dk]   queries with cumulative decay
 *   k_decay_t    [BH, NC, Dk, C]   transposed keys with cumulative decay
 *   dl_exp       [BH, NC, 1, 1]    per-chunk state decay scalar
 *   L_inv        [BH, NC, C, 32]   4 precomputed diagonal block inverses per chunk
 *   initial_state [BH, Dk, Dv]     optional recurrent state (zeros if absent)
 *
 * Returns:
 *   (output [BH, NC, C, Dv], final_state [BH, Dk, Dv])
 */
std::tuple<ttnn::Tensor, ttnn::Tensor> gated_delta_attn_seq(
    const ttnn::Tensor& L_unit,
    const ttnn::Tensor& v_beta_sc,
    const ttnn::Tensor& k_bd_sc,
    const ttnn::Tensor& intra_attn,
    const ttnn::Tensor& q_decay,
    const ttnn::Tensor& k_decay_t,
    const ttnn::Tensor& dl_exp,
    const ttnn::Tensor& L_inv,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    bool token_major_output = false,
    uint32_t num_v_heads = 0,
    uint32_t seq_len = 0);

}  // namespace ttnn::transformer
