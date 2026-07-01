// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ttnn::experimental::apply_twiddles — between-pass elementwise complex
// multiply step of Cooley–Tukey two-pass FFT.  Operates per row:
//
//     out[r, k1] = in[r, k1] * exp(-2*pi*i * (r % N2) * k1 / (N1*N2))
//
// where the input is interpreted as `(M, N1)` with M a multiple of N2.

#pragma once

#include <cstdint>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor> apply_twiddles(
    const ttnn::Tensor& input_real,
    const ttnn::Tensor& input_imag,
    uint32_t N1,
    uint32_t N2);

}  // namespace ttnn::operations::experimental
