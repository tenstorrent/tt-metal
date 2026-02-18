// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Centralize - Compute Kernel
//
// Runs on math RISC-V core (TRISC), performs all compute stages.
//
// Per tile-row (Ht_total iterations), processes 9 sequential phases:
//   Phase 1: tilize(c_0 -> c_1)                 - RM sticks to tiles
//   Phase 2: reduce_row(c_1 -> c_2)              - row mean with scaler 1/W (WaitUpfrontNoPop)
//   Phase 3: sub_col(c_1, c_2 -> c_3)            - x - mean (centered), c_1 popped after
//   Phase 4: square(c_3 -> c_24)                 - c^2 (WaitUpfrontNoPop, c_3 kept)
//   Phase 5: reduce_row(c_24 -> c_4)             - row variance
//   Phase 6: add_scalar(c_4, c_7 -> c_25)        - var + epsilon (c_7 not popped)
//   Phase 7: rsqrt(c_25 -> c_5)                  - 1/sqrt(var + eps)
//   Phase 8: mul_col(c_3, c_5 -> c_6)            - centered * inv_std (c_3 popped)
//   Phase 9: untilize(c_6 -> c_16)               - tiles to RM sticks
//
// Compile-time args:
//   [0]  Wt              - tiles per tile-row
//   [1]  Ht_total        - total tile-rows
//   [2]  cb_rm_in        - c_0
//   [3]  cb_tilized      - c_1
//   [4]  cb_mean         - c_2
//   [5]  cb_centered     - c_3
//   [6]  cb_squared      - c_24
//   [7]  cb_var          - c_4
//   [8]  cb_var_plus_eps - c_25
//   [9]  cb_inv_std      - c_5
//   [10] cb_result       - c_6
//   [11] cb_rm_out       - c_16
//   [12] cb_eps          - c_7
//   [13] cb_scaler       - c_8
//
// Runtime args: (none - all parameters are compile-time for single-core)

#include "compute_kernel_api.h"

void kernel_main() {
    // TODO: Implement compute kernel
    // Kernel writer agent will implement all 9 phases per tile-row using:
    //   - compute_kernel_lib::tilize (tilize_helpers.hpp)
    //   - compute_kernel_lib::reduce<SUM, REDUCE_ROW> (reduce_helpers_compute.hpp)
    //   - compute_kernel_lib::sub<COL> (binary_op_helpers.hpp)
    //   - compute_kernel_lib::square (binary_op_helpers.hpp)
    //   - compute_kernel_lib::add<SCALAR> (binary_op_helpers.hpp)
    //   - rsqrt_tile (eltwise_unary/rsqrt.h)
    //   - compute_kernel_lib::mul<COL> (binary_op_helpers.hpp)
    //   - compute_kernel_lib::untilize (untilize_helpers.hpp)
}
