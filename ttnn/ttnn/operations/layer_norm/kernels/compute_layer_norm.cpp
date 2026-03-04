// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Compute Kernel
//
// Performs the three-pass layer normalization:
//   Pass 1: Cross-tile accumulation + reduce_row -> mean stored in c_24
//   Pass 2: (x-mean)^2 accumulation + reduce_row + eps + rsqrt -> c_26
//   Pass 3: (x-mean) * rsqrt(var+eps) * gamma + beta -> c_16 (output)
//
// Compile-time args:
//   [0] num_rows_per_core : uint32  Tile-rows assigned to this core (group 1 count)
//   [1] Wt               : uint32  Width in tiles
//   [2] has_gamma        : uint32  1 if gamma tensor is present, 0 otherwise
//   [3] has_beta         : uint32  1 if beta tensor is present, 0 otherwise
//
// Runtime args: none

#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    // TODO: Implement compute kernel
    // Stub: do nothing.
    // Stage 1 will add identity passthrough (copy c_0 -> c_16).
}
