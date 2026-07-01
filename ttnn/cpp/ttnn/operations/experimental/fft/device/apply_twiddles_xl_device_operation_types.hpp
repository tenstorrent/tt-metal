// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attribute / tensor-arg types for ttnn::prim::apply_twiddles_xl — the
// large-modulus elementwise complex multiply used by the three-pass
// composite (fft_three_pass) to apply the between-pass-1-and-2 twiddle.
//
// Unlike ttnn::prim::apply_twiddles, which precomputes a dense
// (twiddle_N2 × twiddle_N1) table (one tile per twiddle row, capping
// twiddle_N2 at 1024 ≈ 4 MB), apply_twiddles_xl computes each twiddle
// row on-the-fly from a small delta lookup:
//
//     delta[i] = exp(-2πi · i / full_N)            for i ∈ [0, big_modulus)
//     row_phase_idx = (row % big_modulus)
//     tw[row, k]    = delta[row_phase_idx]^k       (computed by recurrence)
//
// Cost: O(big_modulus) host memory + O(P) per-row recurrence on BRISC0.
// Lets us scale to big_modulus = 2^20 (== N1·N2 for N = 2^30 cube-balanced).
//
// Semantics (input shape (..., M, P)):
//   For each row r ∈ [0, M):
//     row_phase_idx = (r % big_modulus)
//     out[r, k] = in[r, k] * exp(-2πi · row_phase_idx · k / full_N)
//   full_N == P · big_modulus for the canonical three-pass call.

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct ApplyTwiddlesXlParams {
    // Row length (= last dim of input).  Pow-2 in [2, 1024].
    uint32_t P = 0;
    // Twiddle row modulus.  Pow-2; the three-pass composite uses
    // big_modulus = N1·N2 which for cube-balanced N up to 2^30 ranges
    // over [2^10, 2^20].
    uint32_t big_modulus = 0;
    // Full-FFT denominator for the angle.  Pow-2.  Three-pass uses
    // full_N = P · big_modulus = N.
    uint32_t full_N = 0;
};

struct ApplyTwiddlesXlTensorArgs {
    Tensor input_real;
    Tensor input_imag;
};

}  // namespace ttnn::experimental::prim
