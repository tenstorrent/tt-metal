// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Runs on math RISC-V core, performs FPU/SFPU operations
//
// Stage 1 stub: minimal implementation - no-op kernel.
// Full implementation will run these phases per tile-row:
//   Phase 1: tilize(cb_in -> cb_tilized, Wt, 1)
//   Phase 2: reduce<SUM,REDUCE_ROW>(cb_tilized, cb_reduce_scaler -> cb_mean, row(Wt))
//             -- WaitUpfrontNoPop so cb_tilized persists for Phase 3
//   Phase 3: sub<COL>(cb_tilized, cb_mean -> cb_centered, row(Wt))
//             -- manual cb_pop_front(cb_tilized, Wt) after
//   Phase 4: square<WaitUpfrontNoPop>(cb_centered -> cb_squared, row(Wt))
//             -- WaitUpfrontNoPop so cb_centered persists for Phase 7
//   Phase 5: reduce<SUM,REDUCE_ROW>(cb_squared, cb_reduce_scaler -> cb_var_eps, row(Wt))
//   Phase 6: add<SCALAR>(cb_var_eps, cb_eps -> cb_inv_std, single())
//             -- post_op: rsqrt_tile_init(); rsqrt_tile(dst_idx);
//   Phase 7: mul<COL>(cb_centered, cb_inv_std -> cb_normed, row(Wt))
//             -- manual cb_pop_front(cb_centered, Wt) after
//   Phase 8a: mul<ROW>(cb_normed, cb_gamma -> cb_affine_out, row(Wt))
//   Phase 8b: add<ROW>(cb_affine_out, cb_beta -> cb_normed, row(Wt))  [reuse freed cb]
//   Phase 9: untilize<Wt>(cb_normed -> cb_out, 1)

#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

void kernel_main() {
    // Compile-time args:
    //   [0] Wt          - Tiles per tile-row
    //   [1] has_gamma   - 1 if gamma
    //   [2] has_beta    - 1 if beta

    // Runtime args:
    //   [0] N           - Tile-rows to process on this core

    // Stub: no-op
    // Real implementation will execute 9 phases of layer norm computation.
}
