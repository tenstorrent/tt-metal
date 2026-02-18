// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Centralize - Compute Kernel
//
// TDD Stage: centralize
// Phases 1 (tilize), 2 (reduce mean), 3 (sub col), 9 (untilize) are active.
//
// Per tile-row (Ht_total iterations):
//   Phase 1: tilize(c_0 -> c_1)                   - RM sticks to tiles
//   Phase 2: reduce_row SUM (c_1 -> c_2)          - row mean with 1/W scaler
//   Phase 3: sub_col (c_1, c_2 -> c_3)            - centralize: x - mean
//   Phase 9: untilize(c_3 -> c_16)                - tiles back to RM sticks
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
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Ht_total = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rm_in = get_compile_time_arg_val(2);
    constexpr uint32_t cb_tilized = get_compile_time_arg_val(3);
    constexpr uint32_t cb_mean = get_compile_time_arg_val(4);
    constexpr uint32_t cb_centered = get_compile_time_arg_val(5);
    // [6]-[10] unused in this stage
    constexpr uint32_t cb_rm_out = get_compile_time_arg_val(11);
    // [12] cb_eps unused in this stage
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(13);

    // ========== Hardware startup ==========
    // Span all CB IDs used: c_0 (0) through c_25 (25)
    constexpr uint32_t cb_min = cb_rm_in;  // c_0
    constexpr uint32_t cb_max = 25;        // c_25 is the highest CB allocated
    compute_kernel_hw_startup(cb_min, cb_max);

    // ========== Main loop: per tile-row ==========
    for (uint32_t tr = 0; tr < Ht_total; ++tr) {
        // Phase 1: Tilize (c_0 -> c_1)
        // Helper handles: cb_wait_front(c_0, Wt), tilize_block, cb_pop_front(c_0, Wt),
        //                 cb_reserve_back(c_1, Wt), cb_push_back(c_1, Wt)
        compute_kernel_lib::tilize<cb_rm_in, cb_tilized>(Wt, 1);

        // Phase 2: Reduce row mean (c_1 -> c_2)
        // WaitUpfrontNoPop: waits for Wt tiles in c_1, does NOT pop them (tiles persist for Phase 3)
        // Scaler CB (c_8) has 1/W tile from reader startup, never popped
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: Subtract mean (c_1, c_2 -> c_3)
        // Input A (c_1): NoWaitPopAtEnd - already waited in Phase 2, pop all Wt tiles at end
        // Input B (c_2): WaitAndPopPerTile - wait for 1 mean tile, pop after row (COL broadcast)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 9: Untilize (c_3 -> c_16)
        // Output centered values (x - mean) instead of raw tilized values
        // Helper handles: cb_wait_front(c_3, Wt), untilize, cb_pop_front(c_3, Wt),
        //                 cb_reserve_back(c_16, Wt), cb_push_back(c_16, Wt)
        compute_kernel_lib::untilize<Wt, cb_centered, cb_rm_out>(1);
    }
}
