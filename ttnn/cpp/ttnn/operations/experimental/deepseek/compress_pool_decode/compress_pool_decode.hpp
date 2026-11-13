// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek {

// Fused DeepSeek-V4.1-Flash single-row compressor window update + pool + RMSNorm + optional RoPE.
//
// When window tensors (``gate``, ``win_kv``, ``win_gate``, ``win_slot``) are provided (compress_ratio == 2):
//   - Always writes ``kv`` and ``gate`` into row ``win_slot`` of ``win_kv`` and ``win_gate`` (preserving
//     all other rows of the 32x32 TILE cache).
//   - If ``do_pool`` is false: returns an empty vector.
//   - If ``do_pool`` is true: reads ``k0 = win_kv[prev_slot]``, ``g0 = win_gate[prev_slot]`` (where
//     ``prev_slot`` defaults to ``1 - win_slot`` when omitted), computes
//       w0     = sigmoid(g0 - gate)
//       pooled = kv + w0 * (k0 - kv)
//     followed by RMSNorm with ``norm_weight`` and ``epsilon``.
//
// When window tensors are omitted (compress_ratio == 1, ``do_pool == true``):
//   - Applies RMSNorm directly to ``kv`` with ``norm_weight`` and ``epsilon``.
//
// Returns:
//   - ``[]`` when ``do_pool`` is false.
//   - ``[latent]`` (``[1, 1, 1, Dh]`` bfloat16 32x32 TILE DRAM) when ``cos`` and ``sin`` are omitted.
//   - ``[latent, roped_row]`` (where ``roped_row`` is ``[1, 1, 1, Dh]`` bfloat16 ROW_MAJOR DRAM with
//     interleaved RoPE applied to its trailing ``rope_dim`` channels) when ``cos`` and ``sin`` are provided.
std::vector<Tensor> compress_pool_decode(
    const Tensor& kv,
    const Tensor& norm_weight,
    float epsilon = 1e-6f,
    const std::optional<Tensor>& gate = std::nullopt,
    const std::optional<Tensor>& win_kv = std::nullopt,
    const std::optional<Tensor>& win_gate = std::nullopt,
    const std::optional<Tensor>& win_slot = std::nullopt,
    const std::optional<Tensor>& prev_slot = std::nullopt,
    const std::optional<Tensor>& cos = std::nullopt,
    const std::optional<Tensor>& sin = std::nullopt,
    bool do_pool = true);

}  // namespace ttnn::experimental::deepseek
