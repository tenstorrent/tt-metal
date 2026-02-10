// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"

// Compile-time arguments:
// 0: Wt (number of tiles along W dimension)
// 1: num_tile_rows (total number of tile-rows to process)
// 2: has_gamma (1 if gamma is provided, 0 if not)
// 3: has_beta (1 if beta is provided, 0 if not)

// Runtime arguments: None

void kernel_main() {
    // This is a stub compute kernel.
    // Real implementation will:
    //
    // Phase 0 (one-time initialization):
    // 1. Tilize gamma from CB 2 -> CB 3 (if has_gamma)
    // 2. Tilize beta from CB 4 -> CB 5 (if has_beta)
    //
    // Phase 1 (per tile-row loop):
    // 1. Tilize input from CB 0 -> CB 1
    // 2. Reduce SUM with scaler (CB 6) to compute mean: CB 1 -> CB 24
    // 3. Subtract mean (broadcast): CB 1, CB 24 -> CB 25
    // 4. Square centered values: CB 25 -> CB 26
    // 5. Reduce SUM with scaler to compute variance: CB 26 -> CB 27
    // 6. Add epsilon (scalar): CB 27, CB 7 -> CB 27 (reuse)
    // 7. Apply rsqrt: CB 27 -> CB 28
    // 8. Multiply centered by rstd (broadcast): CB 25, CB 28 -> CB 29
    // 9. Multiply by gamma (element-wise): CB 29, CB 3 -> CB 30 (if has_gamma, else skip)
    // 10. Add beta (element-wise): CB 30, CB 5 -> CB 8 (if has_beta, adjust source CB)
    // 11. Untilize output: CB 8 -> CB 16
    //
    // For now, this stub does nothing to allow compilation.
}
