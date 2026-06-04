// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

enum class FFTPrecision : uint8_t {
    Precise = 0,    // SFPU true-fp32 (default; matches torch precision)
    Fast    = 1,    // FPU bf16-mantissa matmul (faster, ~1e-3 round-trip)
};

// Operation-level attributes (kernel-affecting only — see compute_program_hash).
struct FFTParams {
    bool         inverse   = false;
    FFTPrecision precision = FFTPrecision::Precise;
};

// Tensor inputs to the device op. Forward FFT uses input_real only; IFFT
// also requires input_imag (the imaginary half of the spectrum). Carrying
// an optional through the device-op layer keeps the dispatch single-path.
struct FFTTensorArgs {
    Tensor                input_real;
    std::optional<Tensor> input_imag;
};

// Backend selected at validate time, used by the program factory to pick
// which kernel pipeline to instantiate.
enum class FFTBackend : uint8_t {
    Stockham,        // Float32, pow2 N, N <= 1M
    UniversalXL,     // Float32, pow2 N, 1M < N <= 16M
    Universal,       // Float32, non-pow2 N (mixed-radix / Bluestein)
    UniversalBf16,   // BFloat16, any N (true-bf16 FPU matmul)
};

}  // namespace ttnn::experimental::prim
