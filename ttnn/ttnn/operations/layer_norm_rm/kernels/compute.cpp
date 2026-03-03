// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel (STUB)
//
// Runs on TRISC math cores, performs FPU/SFPU operations.
//
// Responsibilities:
//   One-time setup:
//     - Tilize CB_GAMMA_RM (c_6) -> CB_GAMMA (c_4)  [if has_gamma]
//     - Tilize CB_BETA_RM  (c_7) -> CB_BETA  (c_5)  [if has_beta]
//
//   Per-tile-row loop (num_tile_rows iterations):
//     Phase 1: tilize  c_0 -> c_1  (RM sticks -> tilized tiles)
//     Phase 2: reduce<SUM,ROW> c_1+c_2 -> c_24  (mean = sum(x)/W)
//     Phase 3: sub<COL> c_1, c_24 -> c_25  (centered = x - mean)
//     Phase 4: square   c_25 -> c_27  (squared = centered^2, c_25 persists)
//     Phase 5: reduce<SUM,ROW> c_27+c_2 -> c_26  (variance = sum(sq)/W)
//     Phase 5b: add(eps) + rsqrt: c_26+c_3 -> c_24  (inv_std = rsqrt(var+eps))
//     Phase 6: mul<COL> c_25, c_24 -> c_27  (normed = centered * inv_std)
//     Phase 7: mul<NONE> c_27, c_4 -> c_28  (gamma * normed)  [if has_gamma]
//     Phase 8: add<NONE> c_28, c_5 -> c_27  (+ beta)          [if has_beta]
//     Phase 9: untilize c_27 -> c_16  (normed tiles -> RM sticks)
//
// Compile-time args:
//   [0] Wt         : tiles per row (W / 32)
//   [1] has_gamma  : 1 if gamma tensor present
//   [2] has_beta   : 1 if beta tensor present

#include "api/compute/common.h"

void kernel_main() {
    // STUB: No-op implementation.
    // The kernel-writer TDD agent will implement each phase using
    // compute_kernel_lib helpers (tilize, reduce, sub, square, mul, add,
    // untilize) following the phase sequence in op_design.md Part 2.
}
