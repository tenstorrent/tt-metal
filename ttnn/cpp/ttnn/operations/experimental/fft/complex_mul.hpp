// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ttnn::experimental::complex_mul — fused ROW_MAJOR elementwise
// complex multiply of two same-shape complex tensors.
//
// Semantics, for A = (a_real, a_imag) and B = (b_real, b_imag):
//
//   out_real = a_real * b_real  -  a_imag * b_imag
//   out_imag = a_real * b_imag  +  a_imag * b_real
//
// All four input tensors must share the same shape, dtype (fp32 or
// bf16), and ROW_MAJOR layout.  Compute is fp32 internally; bf16
// inputs/outputs are expanded/truncated at the kernel I/O boundary.
//
// Used by:
//   - Bluestein composite (commit 6d): chirp pre/post multiply, H
//     multiply (the spectral-domain step of the chirp-z transform).
//   - IFFT inverse path (commit 6c): conjugate-and-scale via a length-1
//     broadcast tensor of (1/N, -1/N).
//
// Constraints:
//   - All tensors same shape, dtype (fp32 or bf16), and ROW_MAJOR layout.
//   - Last-dim (P) row length must be in [1, 1024].

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental {

std::tuple<ttnn::Tensor, ttnn::Tensor> complex_mul(
    const ttnn::Tensor& a_real,
    const ttnn::Tensor& a_imag,
    const ttnn::Tensor& b_real,
    const ttnn::Tensor& b_imag);

}  // namespace ttnn::operations::experimental
