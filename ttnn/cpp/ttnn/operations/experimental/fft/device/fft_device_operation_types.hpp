// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

enum class FFTPrecision : uint8_t {
    Precise = 0,  // SFPU true-fp32 (default; matches torch precision)
    Fast = 1,     // FPU bf16-mantissa matmul (faster, ~1e-3 round-trip)
};

// Operation-level attributes (kernel-affecting only — see compute_program_hash).
struct FFTParams {
    bool inverse = false;
    FFTPrecision precision = FFTPrecision::Precise;
};

// Tensor inputs to the device op. Forward FFT uses input_real only; IFFT
// also requires input_imag (the imaginary half of the spectrum). Carrying
// an optional through the device-op layer keeps the dispatch single-path.
struct FFTTensorArgs {
    Tensor input_real;
    std::optional<Tensor> input_imag;
};

}  // namespace ttnn::experimental::prim
