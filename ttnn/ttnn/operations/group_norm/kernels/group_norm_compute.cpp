// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Group Norm - Compute Kernel (Stub)
// Runs on math RISC-V core, performs tilize/stats/normalize/untilize
//
// Phases:
//   Phase 0: Tilize (cb_input_rm -> cb_tilized, persistent)
//   Phase 1: Reduce Sum -> Mean per group
//   Phase 2: Reduce Sum-of-Squares -> E[x^2] per group
//   Phase 3: Compute Variance -> Den per group
//   Phase 4: Normalize pass (sub mean, mul den)
//   Phase 5: Affine (mul gamma, add beta)
//   Phase 6: Untilize (cb_normalized -> cb_output_rm)

#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    // Stub: kernel body will be implemented by kernel-writer agent.
    // The compute kernel will:
    //   For each sample n in 0..N-1:
    //     1. Tilize Ht tile-rows (each Ct tiles) from cb_input_rm -> cb_tilized
    //     2. Wait for all Ht*Ct tiles in cb_tilized
    //     3. For each group g: compute mean via reduce_tile (Phase 1)
    //     4. For each group g: compute E[x^2] via squared reduce (Phase 2)
    //     5. For each group g: compute den = rsqrt(var + eps) (Phase 3)
    //     6. For each tile-row, for each tile: normalize + affine (Phase 4+5)
    //     7. Untilize each tile-row from cb_normalized -> cb_output_rm (Phase 6)
    //     8. Pop cb_tilized
}
