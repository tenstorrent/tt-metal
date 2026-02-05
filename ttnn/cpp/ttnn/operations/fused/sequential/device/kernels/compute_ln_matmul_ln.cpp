// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Sequential Compute: Add -> Matmul -> Add
 *
 * Simplified sequential test demonstrating:
 * 1. Eltwise add: X + bias1 -> temp1
 * 2. Matmul: temp1 * W -> temp2
 * 3. Eltwise add: temp2 + bias2 -> output
 *
 * This tests the core sequential compute pattern without the complexity
 * of full layernorm (which requires row-wise reduction with helper functions).
 */

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

#include <cstdint>

void kernel_main() {
    // Compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);  // Width in tiles

    // CB assignments
    constexpr auto cb_in = tt::CBIndex::c_0;       // input X [1, Wt]
    constexpr auto cb_bias1 = tt::CBIndex::c_1;    // bias1 [1, Wt]
    constexpr auto cb_weights = tt::CBIndex::c_2;  // weights W [Wt, Wt]
    constexpr auto cb_bias2 = tt::CBIndex::c_3;    // bias2 [1, Wt]
    constexpr auto cb_temp1 = tt::CBIndex::c_4;    // temp after first add [1, Wt]
    constexpr auto cb_temp2 = tt::CBIndex::c_5;    // temp after matmul [1, Wt]
    constexpr auto cb_out = tt::CBIndex::c_16;     // final output [1, Wt]

    // ========================================================================
    // PHASE 1: X + bias1 -> temp1
    // ========================================================================
    binary_op_init_common(cb_in, cb_bias1, cb_temp1);
    add_tiles_init(cb_in, cb_bias1);

    cb_wait_front(cb_in, Wt);
    cb_wait_front(cb_bias1, Wt);
    cb_reserve_back(cb_temp1, Wt);

    for (uint32_t w = 0; w < Wt; w++) {
        tile_regs_acquire();
        add_tiles(cb_in, cb_bias1, w, w, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_temp1);
        tile_regs_release();
    }

    cb_push_back(cb_temp1, Wt);
    cb_pop_front(cb_in, Wt);
    cb_pop_front(cb_bias1, Wt);
    cb_wait_front(cb_temp1, Wt);

    // ========================================================================
    // PHASE 2: Matmul - temp1[1, Wt] * W[Wt, Wt] = temp2[1, Wt]
    // ========================================================================
    mm_init(cb_temp1, cb_weights, cb_temp2);

    // For each output column
    for (uint32_t nt = 0; nt < Wt; nt++) {
        acquire_dst();

        // Accumulate across the inner dimension
        for (uint32_t kt = 0; kt < Wt; kt++) {
            cb_wait_front(cb_weights, 1);
            matmul_tiles(cb_temp1, cb_weights, kt, 0, 0);
            cb_pop_front(cb_weights, 1);
        }

        cb_reserve_back(cb_temp2, 1);
        pack_tile(0, cb_temp2);
        cb_push_back(cb_temp2, 1);

        release_dst();
    }

    cb_pop_front(cb_temp1, Wt);
    cb_wait_front(cb_temp2, Wt);

    // ========================================================================
    // PHASE 3: temp2 + bias2 -> output
    // ========================================================================
    binary_op_init_common(cb_temp2, cb_bias2, cb_out);
    add_tiles_init(cb_temp2, cb_bias2);

    cb_wait_front(cb_bias2, Wt);
    cb_reserve_back(cb_out, Wt);

    for (uint32_t w = 0; w < Wt; w++) {
        tile_regs_acquire();
        add_tiles(cb_temp2, cb_bias2, w, w, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }

    cb_push_back(cb_out, Wt);
    cb_pop_front(cb_temp2, Wt);
    cb_pop_front(cb_bias2, Wt);
}
