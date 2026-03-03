// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel (stub)
// Runs on math RISC-V core (TRISC), performs tile operations.
//
// Compile-time args:
//   [0] num_rows   - total tile-rows to process (N_outer * Ht)
//   [1] Wt         - tiles per row (W / 32)
//
// CB usage (all tile-sized pages):
//   c_0  (cb_input_rm)    - consume Wt RM pages per row (from reader)
//   c_1  (cb_gamma_rm)    - consume Wt RM pages once
//   c_2  (cb_beta_rm)     - consume Wt RM pages once
//   c_8  (cb_scaler)      - 1 tile: 1/W reduce scaler (never popped)
//   c_9  (cb_eps)         - 1 tile: epsilon scaler (never popped)
//   c_16 (cb_out_rm)      - produce Wt RM pages per row (to writer)
//   c_24 (cb_input_tiled) - tilized input row (Wt tiles)
//   c_25 (cb_mean)        - 1 mean tile (col0 valid)
//   c_26 (cb_centered)    - x - mean (Wt tiles); reused for gamma output
//   c_27 (cb_var_sq)      - centered^2 (Wt tiles)
//   c_28 (cb_inv_std)     - 1 inv_std tile (col0 valid)
//   c_29 (cb_gamma_tiled) - tilized gamma (Wt tiles, program lifetime)
//   c_30 (cb_beta_tiled)  - tilized beta  (Wt tiles, program lifetime)
//   c_31 (cb_normed)      - x_centered * inv_std (Wt tiles)
//
// Phases per row:
//   Setup: tilize c_1->c_29 (gamma), tilize c_2->c_30 (beta)
//   0: tilize   c_0  -> c_24
//   1: reduce   c_24, c_8 -> c_25   (mean, WaitUpfrontNoPop on c_24)
//   2: sub(COL) c_24, c_25 -> c_26  (centered; pop c_24 and c_25 after)
//   3: square   c_26 -> c_27        (WaitUpfrontNoPop on c_26)
//   4: reduce   c_27, c_8 -> c_28 + rsqrt (var+eps -> inv_std)
//   5: mul(COL) c_26, c_28 -> c_31  (pop c_26 and c_28 after)
//   6: mul(ROW) c_31, c_29 -> c_26  (normed * gamma; NoWaitNoPop on c_29)
//   7: add(ROW) c_26, c_30 -> c_24  (+ beta; NoWaitNoPop on c_30; reuse c_24)
//   8: untilize c_24 -> c_16

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

void kernel_main() {
    // Stub: no-op
    // Real implementation performs 9 phases of computation per tile-row.
}
