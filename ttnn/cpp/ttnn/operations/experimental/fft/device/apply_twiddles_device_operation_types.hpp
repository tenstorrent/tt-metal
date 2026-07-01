// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attribute / tensor-arg types for ttnn::prim::apply_twiddles — the
// between-pass elementwise complex-multiply step of Cooley–Tukey two-pass
// FFT.  Lives in its own header (NOT in fft_device_operation_types.hpp)
// because it is a standalone op that the FFT composite will call; the FFT
// op-types stay focused on the user-visible FFT API.

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Operation-level attributes (kernel-affecting only — included in the
// program hash).  N is implicit: N == N1 * N2 derived from these.
struct ApplyTwiddlesParams {
    uint32_t N1 = 0;   // inner pass-1 length (one row of the input tensor)
    uint32_t N2 = 0;   // outer pass-2 length (twiddle modulus)
};

// Tensor inputs: real + imag of the intermediate (post-Pass-1) signal.
// Last dim of both must equal N1; product of leading dims must be a
// multiple of N2.
struct ApplyTwiddlesTensorArgs {
    Tensor input_real;
    Tensor input_imag;
};

}  // namespace ttnn::experimental::prim
