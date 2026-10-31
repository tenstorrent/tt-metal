// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::mhc_pre_ttnn {

// ttnn.bringup.mhc_pre_xing: the coefficient + collapse half of mhc_pre for a caller that has already reduced the
// projection (Xing4.0 mHC, xing40_a4b_d_p P.2b; a separate entry, mhc_pre itself is unchanged).
//
// coefficients_given = false: `input` is the all-reduced row (..., T, 32) fp32 = [mix 0 .. n(n+2)-1 | sum x^2 | 0 ..]
//   (the unnormalised projection x @ fn^T and sum x^2 over the full n*H width). Per token:
//     r = rsqrt(sum x^2 / norm_width + norm_eps); z = scale_g * mix * r + base
//     pre = sigmoid(z[0:n]); post = 2 sigmoid(z[n:2n]); L = clamp(z[2n:], clamp_min, clamp_max) (L[i][j] = [i n + j])
//     comb = exp(L - rowmax); sinkhorn_iters x { m / (rowsum + hc_eps), then m / (colsum + hc_eps) }
//   -> hc (..., T, n(n+2)) fp32 = [pre | post | comb row-major].
// deepseek = true (default false): DeepSeek-V4's math (inference/kernel.py hc_split_sinkhorn_kernel): pre = sigmoid(z)
//   + hc_eps; L not clamped (clamp_min / clamp_max ignored); comb = exp(L - rowmax) / rowsum + hc_eps, then
//   m / (colsum + hc_eps), then (sinkhorn_iters - 1) x { rows, columns } as above. Needs the coefficients.
// coefficients_given = true: `input` is a finished hc (..., T, n(n+2)) fp32 (only pre, columns 0..n-1, is read).
// streams (..., T, n*C) fp32, optional: y = sum_i pre_i * streams[..., i*C:(i+1)*C] (..., T, C) fp32.
// collapse_pre (..., T, n..32) fp32, optional (default none; needs streams and the coefficients): y collapses with
//   collapse_pre's columns 0..n-1 instead of the pre just computed (single-pass mHC, deepseek_v41_flash_d_p P.14: a
//   sublayer collapses with the previous sublayer's pre); hc is still returned. Compile define MHC_XING_GIVEN_PRE
//   (absent when off: the program is the one it was).
// Returns (hc or None, y or None): hc when the coefficients are computed, y when streams are given.
std::tuple<std::optional<Tensor>, std::optional<Tensor>> mhc_pre_xing(
    const Tensor& input,
    const std::optional<Tensor>& streams,
    const std::vector<double>& scale,
    const std::vector<double>& base,
    double norm_width,
    int64_t n = 4,
    double norm_eps = 1e-6,
    double hc_eps = 1e-6,
    int64_t sinkhorn_iters = 20,
    double clamp_min = -30.0,
    double clamp_max = 30.0,
    bool coefficients_given = false,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config = std::nullopt,
    bool deepseek = false,
    const std::optional<Tensor>& collapse_pre = std::nullopt,
    int64_t partials = 1,
    const std::optional<Tensor>& hc_output = std::nullopt,
    const std::optional<Tensor>& y_output = std::nullopt,
    std::optional<tt::tt_metal::CoreCoord> max_grid = std::nullopt);

// ttnn.bringup.mhc_pre_xing_pack: the row mhc_pre_xing reduces, from the caller's partial projection. `mix` (..., T,
// 32) fp32 (partial x @ fn^T in columns 0 .. n(n+2)-1, zero elsewhere) and the local streams (..., T, n*C) fp32 -> mix
// with column n (n + 2) = sum over the row of streams^2 (exact fp32, SFPU), one pass over the streams.
// Engram mode (engram_keys and engram_weight given, mix None; deepseek_v41_flash_d_p P.14): a separate program, one
// pass over the streams h and the keys k (engram_keys (..., T, >= n*C) fp32, the keys in columns 0 .. n*C - 1) with
// the weight w (engram_weight (..., 32, n*C) fp32, the weight row on all 32 rows) -> a new row (..., T, 32) fp32 =
// [sum_c h_i w_i k_i (i < n) | sum_c h_i^2 | sum_c k_i^2 | 0 ...] (exact fp32 SFPU; the engram gate's partials).
// engram_gate_eps (engram mode only, deepseek_v41_flash_d_p P.24; default none = off): the C columns are the whole
// hidden, and columns 3n .. 4n-1 also hold the gates sigmoid(copysign(sqrt(max(|d_i|, 1e-6)), d_i)),
// d_i = dot_i rsqrt(sum h_i^2 / C + eps) rsqrt(sum k_i^2 / C + eps) / sqrt(C) (fp32 SFPU, the op chain's steps).
Tensor mhc_pre_xing_pack(
    const std::optional<Tensor>& mix,
    const Tensor& streams,
    int64_t n = 4,
    const std::optional<Tensor>& engram_keys = std::nullopt,
    const std::optional<Tensor>& engram_weight = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    std::optional<tt::tt_metal::CoreCoord> max_grid = std::nullopt,
    std::optional<float> engram_gate_eps = std::nullopt);

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn
