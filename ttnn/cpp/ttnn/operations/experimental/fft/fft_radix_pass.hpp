// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ttnn::experimental::fft_radix_pass — fused [batched length-P FFT
// + optional post-twiddle complex multiply].  Single dispatch.
// Building block for the K-pass composite (Cooley–Tukey universal-XL).
//
// Semantics, for input of shape (..., M, P):
//   out_complex[r, k] = FFT_P(in_complex[r, :])[k]
//   if twiddle_N2 != 0:
//     row_idx = (r / stride) % twiddle_N2           (stride defaults to 1)
//     out_complex[r, k] *= exp(-2πi · row_idx · k / (P · twiddle_N2))
//
// Constraints:
//   - P pow-2 in [2, 1024]
//   - product of leading dims pow-2 and >= 1
//   - twiddle_N2 either 0 (no PT) or pow-2 in [1, 1024]
//   - stride pow-2 in [1, M] dividing M, and (M / stride) % twiddle_N2 == 0
//   - fp32 or bf16; ROW_MAJOR layout

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor> fft_radix_pass(
    const ttnn::Tensor& input_real,
    const std::optional<ttnn::Tensor>& input_imag,
    uint32_t P,
    uint32_t twiddle_N2,
    uint32_t stride       = 1,
    float    output_scale = 1.0f);

}  // namespace ttnn::operations::experimental
