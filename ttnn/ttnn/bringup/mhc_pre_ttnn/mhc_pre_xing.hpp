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
// coefficients_given = true: `input` is a finished hc (..., T, n(n+2)) fp32 (only pre, columns 0..n-1, is read).
// streams (..., T, n*C) fp32, optional: y = sum_i pre_i * streams[..., i*C:(i+1)*C] (..., T, C) fp32.
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
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config = std::nullopt);

// ttnn.bringup.mhc_pre_xing_pack: the row mhc_pre_xing reduces, from the caller's partial projection. `mix` (..., T,
// 32) fp32 (partial x @ fn^T in columns 0 .. n(n+2)-1, zero elsewhere) and the local streams (..., T, n*C) fp32 -> mix
// with column n (n + 2) = sum over the row of streams^2 (exact fp32, SFPU), one pass over the streams.
Tensor mhc_pre_xing_pack(const Tensor& mix, const Tensor& streams, int64_t n = 4);

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn
