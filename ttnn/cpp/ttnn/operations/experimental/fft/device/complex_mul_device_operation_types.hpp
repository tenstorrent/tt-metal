// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attribute / tensor-arg types for ttnn::prim::complex_mul — fused
// ROW_MAJOR elementwise complex multiply.  Building block for the
// Bluestein (chirp-z) composite (commit 6d) and for the IFFT
// conjugate-and-scale step (commit 6c).
//
// Semantics, for two same-shape complex tensors A = (a_re, a_im) and
// B = (b_re, b_im):
//
//   out_re[r, k] = a_re[r, k] * b_re[r, k]  -  a_im[r, k] * b_im[r, k]
//   out_im[r, k] = a_re[r, k] * b_im[r, k]  +  a_im[r, k] * b_re[r, k]
//
// All operands and the output share the same shape, dtype (fp32 or
// bf16), and ROW_MAJOR layout.  Compute is fp32 internally; bf16
// inputs/outputs are expanded/truncated at the kernel I/O boundary
// (same policy as apply_twiddles).
//
// Implementation note: the compute and writer kernels are reused
// VERBATIM from apply_twiddles (CB layout is identical — A in
// CB_A_R/A_I, B in CB_T_R/T_I, output in CB_B_R/B_I).  Only the reader
// is new: it loads both A and B tiles from DRAM instead of building B
// on the fly from a delta lookup.

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// No kernel-affecting attributes — the only knobs are dtype + shape,
// both derived from the input tensors.  An empty struct keeps the
// device-op machinery happy (hash + program-cache key paths still
// expect an attributes object to exist).
struct ComplexMulParams {};

struct ComplexMulTensorArgs {
    Tensor a_real;
    Tensor a_imag;
    Tensor b_real;
    Tensor b_imag;
};

}  // namespace ttnn::experimental::prim
