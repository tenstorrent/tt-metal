// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
    // Whether the caller supplied an imaginary input. Lives here rather than
    // alongside the tensors because tensor_args is walked by the reflection
    // visitor, which only handles tensor-like leaves.
    bool input_imag_provided = false;
};

// Tensor inputs to the device op. The public optional imaginary input is
// resolved to a real tensor before launch so ProgramSpec sees only owned,
// adapter-visible tensor bindings.
struct FFTTensorArgs {
    Tensor input_real;
    // Effective imaginary input. For real-only calls this is the cached zero
    // tensor, making every ProgramSpec tensor binding visible to the adapter.
    Tensor input_imag;
    Tensor tw_real;
    Tensor tw_imag;
};

}  // namespace ttnn::experimental::prim
