// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Sequential Eltwise Compute Kernel
 *
 * This kernel demonstrates running multiple elementwise operations sequentially
 * on the same core. Each phase runs to completion before the next starts.
 *
 * Phase 0: A + B → scratch1
 * Phase 1: scratch1 * C → scratch2
 * Phase 2: scratch2 + D → output
 *
 * Uses two scratch buffers (ping-pong) to avoid read/write conflicts when
 * processing multiple tiles.
 */

#include "compute_kernel_api/eltwise_binary.h"

#include <cstdint>

#include "compute_kernel_api/tile_move_copy.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t num_phases = get_arg_val<uint32_t>(1);

    // CBs for phase 0: A + B
    constexpr auto cb_in0 = tt::CBIndex::c_0;       // input A
    constexpr auto cb_in1 = tt::CBIndex::c_1;       // input B
    constexpr auto cb_scratch1 = tt::CBIndex::c_2;  // scratch space 1
    constexpr auto cb_scratch2 = tt::CBIndex::c_5;  // scratch space 2 (for ping-pong)

    // CBs for phase 1: scratch * C
    constexpr auto cb_in2 = tt::CBIndex::c_3;  // input C

    // CBs for phase 2: scratch + D
    constexpr auto cb_in3 = tt::CBIndex::c_4;   // input D
    constexpr auto cb_out = tt::CBIndex::c_16;  // final output

    // ========== PHASE 0: A + B → scratch1 ==========
    binary_op_init_common(cb_in0, cb_in1, cb_scratch1);
    add_tiles_init(cb_in0, cb_in1);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        cb_reserve_back(cb_scratch1, 1);

        tile_regs_acquire();
        add_tiles(cb_in0, cb_in1, 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_scratch1);
        tile_regs_release();

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        cb_push_back(cb_scratch1, 1);
    }

    if (num_phases == 1) {
        // Single phase: copy scratch1 to output
        for (uint32_t i = 0; i < num_tiles; ++i) {
            cb_wait_front(cb_scratch1, 1);
            cb_reserve_back(cb_out, 1);

            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_scratch1);
            copy_tile(cb_scratch1, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();

            cb_pop_front(cb_scratch1, 1);
            cb_push_back(cb_out, 1);
        }
        return;
    }

    // ========== PHASE 1: scratch1 * C → scratch2 ==========
    // Re-init for multiply, use separate CBs to avoid read/write conflicts
    binary_op_init_common(cb_scratch1, cb_in2, cb_scratch2);
    mul_tiles_init(cb_scratch1, cb_in2);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_scratch1, 1);
        cb_wait_front(cb_in2, 1);
        cb_reserve_back(cb_scratch2, 1);

        tile_regs_acquire();
        mul_tiles(cb_scratch1, cb_in2, 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_scratch2);
        tile_regs_release();

        cb_pop_front(cb_scratch1, 1);
        cb_pop_front(cb_in2, 1);
        cb_push_back(cb_scratch2, 1);
    }

    if (num_phases == 2) {
        // Two phases: copy scratch2 to output
        for (uint32_t i = 0; i < num_tiles; ++i) {
            cb_wait_front(cb_scratch2, 1);
            cb_reserve_back(cb_out, 1);

            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_scratch2);
            copy_tile(cb_scratch2, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();

            cb_pop_front(cb_scratch2, 1);
            cb_push_back(cb_out, 1);
        }
        return;
    }

    // ========== PHASE 2: scratch2 + D → output ==========
    // Re-init for add
    binary_op_init_common(cb_scratch2, cb_in3, cb_out);
    add_tiles_init(cb_scratch2, cb_in3);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_scratch2, 1);
        cb_wait_front(cb_in3, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        add_tiles(cb_scratch2, cb_in3, 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_pop_front(cb_scratch2, 1);
        cb_pop_front(cb_in3, 1);
        cb_push_back(cb_out, 1);
    }
}
