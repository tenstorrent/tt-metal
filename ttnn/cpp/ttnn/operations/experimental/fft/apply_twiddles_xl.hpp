// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ttnn::experimental::apply_twiddles_xl — large-modulus between-pass
// elementwise complex multiply used by the fft_three_pass composite
// (commit 5).  Unlike ttnn::experimental::apply_twiddles (which caps at
// twiddle_N2 ≤ 1024 due to a precomputed twiddle table), this op builds
// each twiddle row on-the-fly from a small per-(device, big_modulus,
// full_N) delta lookup, letting big_modulus scale to 2^20.
//
// Semantics, for input of shape (..., M, P):
//   row_phase_idx = (row % big_modulus)
//   out[r, k] = in[r, k] · exp(-2πi · row_phase_idx · k / full_N)
//
// Constraints:
//   - P pow-2 in [2, 1024]
//   - big_modulus pow-2 in [1, 2^20]; M must be a multiple of big_modulus
//   - full_N pow-2 and >= big_modulus
//   - fp32 or bf16; ROW_MAJOR layout

#pragma once

#include <cstdint>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor> apply_twiddles_xl(
    const ttnn::Tensor& input_real,
    const ttnn::Tensor& input_imag,
    uint32_t P,
    uint32_t big_modulus,
    uint32_t full_N);

}  // namespace ttnn::operations::experimental
