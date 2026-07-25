// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental {

// Per-tile forward-substitution triangle solve of  L X = RHS  for a single 32x32 tile.
//
// L is a NEGATED unit-lower-triangular matrix: the caller pre-negates the strict-lower entries
// (the diagonal is an implicit 1 and the upper triangle is ignored by the kernel). The solve is
// done on the SFPU via the wired-in compute API `triangle_solve_tile`.
//
// Args (both device, TILE layout, bfloat16, logical shape [1, 1, 32, 32]):
//   l_neg : negated unit-lower-triangular L
//   rhs   : right-hand-side matrix
//
// Returns:
//   x     : [1, 1, 32, 32] bf16 TILE — the solution of  L X = RHS
ttnn::Tensor triangle_solve(
    const ttnn::Tensor& l_neg,
    const ttnn::Tensor& rhs,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental
