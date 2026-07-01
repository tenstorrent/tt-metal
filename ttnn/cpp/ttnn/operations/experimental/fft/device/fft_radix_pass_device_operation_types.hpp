// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attribute / tensor-arg types for ttnn::prim::fft_radix_pass — the
// fused [batched length-P FFT  +  optional post-twiddle cmul] kernel
// that is the building block for the K-pass composite (commit 5,
// fft_universal_xl for N up to 1G).
//
// Semantics, for input of shape (..., M, P):
//   For each row r ∈ [0, M):
//     y[r, :] = FFT_P(in[r, :])
//     if apply_post_twiddle:
//       row_idx = (r / stride) % twiddle_N2          (stride defaults to 1)
//       y[r, k] *= exp(-2πi * row_idx * k / (P * twiddle_N2))
//
// twiddle_N1 is implicit and always equals P.
//
// `stride` lets the three-pass composite (commit 5) reuse fft_radix_pass
// for its Pass-2 step without an extra transpose: with rows enumerating
// (b, n1, k3) at stride N3, setting stride=N3, twiddle_N2=N1 picks the
// correct twiddle row n1 = (b*N1 + n1) % N1.

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct FftRadixPassParams {
    // FFT length per row.  Pow-2 in [2, 1024].
    uint32_t P = 0;
    // 0 → no post-twiddle (pure batched FFT, same observable behaviour
    //     as a BatchedStockhamFactory call).
    // >0 → multiply each row's FFT output by twiddle row
    //     (r / stride) % twiddle_N2.  Pow-2 in [1, 1024] and must divide
    //     the product of leading dims of the input (after applying stride).
    uint32_t twiddle_N2 = 0;
    // Row-index stride for the post-twiddle lookup.  Must be a pow-2 in
    //     [1, M] that divides M (i.e. (M / stride) % twiddle_N2 == 0).
    //     Default 1 = "use r directly" (commit 4 / two-pass behaviour).
    uint32_t stride = 1;
    // Output scalar (added commit 6c, for IFFT).  When != 1.0f, the
    // writer multiplies each STATE element by this value after the
    // (optional) post-twiddle, before the bf16 truncation / DRAM write.
    // The composite's `inverse` flag sets this to 1/N on the LAST
    // radix_pass call to fold the IFFT 1/N scale into the FFT chain
    // with zero extra dispatch.
    //
    // Program-cache identity: the runtime float value does NOT affect
    // the kernel binary, but the BOOLEAN "is_scale_enabled" does (it
    // controls whether the writer compiles in the per-element multiply
    // loop).  We hash on the boolean only.
    float output_scale = 1.0f;
};

// input_imag is optional: for a Pass-1 (real input) radix pass we leave
// it empty and the factory wires up a cached zero scratch.  For Pass-2
// (complex input) the caller passes the imag tensor.
struct FftRadixPassTensorArgs {
    Tensor                input_real;
    std::optional<Tensor> input_imag;
};

}  // namespace ttnn::experimental::prim
