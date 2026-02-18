// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Centralize - Compute Kernel
//
// TDD Stage: tilize_untilize
// Only Phase 1 (tilize) and Phase 9 (untilize) are active.
// For this stage, untilize reads from c_1 (cb_tilized) instead of c_6 (cb_result).
//
// Per tile-row (Ht_total iterations):
//   Phase 1: tilize(c_0 -> c_1)   - RM sticks to tiles
//   Phase 9: untilize(c_1 -> c_16) - tiles back to RM sticks (identity roundtrip)
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

#include "compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Ht_total = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rm_in = get_compile_time_arg_val(2);
    constexpr uint32_t cb_tilized = get_compile_time_arg_val(3);
    // [4]-[10] unused in this stage
    constexpr uint32_t cb_rm_out = get_compile_time_arg_val(11);
    // [12]-[13] unused in this stage

    // ========== Hardware startup ==========
    // Span all CB IDs used: c_0 (0) through c_25 (25)
    // Even though this stage only uses c_0, c_1, c_16, we must cover the range
    // for CBs that are allocated (the hw_startup just initializes pack/unpack config)
    constexpr uint32_t cb_min = cb_rm_in;  // c_0
    constexpr uint32_t cb_max = 25;        // c_25 is the highest CB allocated
    compute_kernel_hw_startup(cb_min, cb_max);

    // ========== Main loop: per tile-row ==========
    for (uint32_t tr = 0; tr < Ht_total; ++tr) {
        // Phase 1: Tilize (c_0 -> c_1)
        // Helper handles: cb_wait_front(c_0, Wt), tilize_block, cb_pop_front(c_0, Wt),
        //                 cb_reserve_back(c_1, Wt), cb_push_back(c_1, Wt)
        compute_kernel_lib::tilize<cb_rm_in, cb_tilized>(Wt, 1);

        // Phase 9: Untilize (c_1 -> c_16)
        // For this TDD stage, untilize reads from c_1 (cb_tilized) instead of c_6 (cb_result)
        // Helper handles: cb_wait_front(c_1, Wt), untilize, cb_pop_front(c_1, Wt),
        //                 cb_reserve_back(c_16, Wt), cb_push_back(c_16, Wt)
        compute_kernel_lib::untilize<Wt, cb_tilized, cb_rm_out>(1);
    }
}
