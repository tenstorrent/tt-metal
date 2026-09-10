// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ttnn::experimental::transpose_rm — fp32/bf16-preserving inner-axis
// transpose for ROW_MAJOR tensors.  Unlike ttnn::transpose which
// internally downcasts to bf16 for ROW_MAJOR fp32, this op preserves
// full-precision data (it's pure data movement, no math).

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental {

// Swap the last two dims of `input` (shape (..., A, C) → (..., C, A)).
// Constraints: A, C must be multiples of 32 and >= 32; fp32 or bf16;
// ROW_MAJOR layout.
ttnn::Tensor transpose_rm(const ttnn::Tensor& input);

}  // namespace ttnn::operations::experimental
