// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Ternary SFPU compute kernel with optional ROW broadcast via LLK unary_bcast
// Supports broadcasting 0, 1 or 2 inputs (A/B/C)
// Expects reader kernel to not perform per-tile fill (FILL_TILE_WITH_FIRST_ROW) controlled by BCAST_LLK flag
//
// The actual ternary operation is provided via:
//   TERNARY_SFPU_OP_INIT()
//   TERNARY_SFPU_OP_FUNC(src_idx_a, src_idx_b, src_idx_c, dst_idx_out)
// which are configured by the program factory for the desired op (e.g., where).

#include <cstdint>

#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/snake_beta.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // typically 1

    // Pre-CBs for inputs A, B, C and output CB
    CircularBuffer cb_out(tt::CBIndex::c_3);
#if BCAST_A
    CircularBuffer cb_pre_a(tt::CBIndex::c_0);
    CircularBuffer cb_bcast_a(tt::CBIndex::c_4);
    CircularBuffer cb_eff_a = cb_bcast_a;
#else
    CircularBuffer cb_eff_a(tt::CBIndex::c_0);
#endif
#if BCAST_B
    CircularBuffer cb_pre_b(tt::CBIndex::c_1);
    CircularBuffer cb_bcast_b(tt::CBIndex::c_5);
    CircularBuffer cb_eff_b = cb_bcast_b;
#else
    CircularBuffer cb_eff_b(tt::CBIndex::c_1);
#endif
#if BCAST_C
    CircularBuffer cb_pre_c(tt::CBIndex::c_2);
    CircularBuffer cb_bcast_c(tt::CBIndex::c_6);
    CircularBuffer cb_eff_c = cb_bcast_c;
#else
    CircularBuffer cb_eff_c(tt::CBIndex::c_2);
#endif

    // Initialize pack/format for SFPU-style ternary kernels (matches existing ternary SFPU kernels)
    unary_op_init_common(cb_eff_a.get_cb_id(), cb_out.get_cb_id());

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
#if BCAST_A
        {
            cb_pre_a.wait_front(num_tiles_per_cycle);
            cb_bcast_a.reserve_back(num_tiles_per_cycle);
            unary_bcast_init<BroadcastType::ROW>(cb_pre_a.get_cb_id(), cb_bcast_a.get_cb_id());

            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(cb_pre_a.get_cb_id(), 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_bcast_a.get_cb_id());
            cb_bcast_a.push_back(num_tiles_per_cycle);
            tile_regs_release();

            cb_pre_a.pop_front(num_tiles_per_cycle);
        }
#endif

#if BCAST_B
        {
            cb_pre_b.wait_front(num_tiles_per_cycle);
            cb_bcast_b.reserve_back(num_tiles_per_cycle);
            unary_bcast_init<BroadcastType::ROW>(cb_pre_b.get_cb_id(), cb_bcast_b.get_cb_id());

            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(cb_pre_b.get_cb_id(), 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_bcast_b.get_cb_id());
            cb_bcast_b.push_back(num_tiles_per_cycle);
            tile_regs_release();

            cb_pre_b.pop_front(num_tiles_per_cycle);
        }
#endif

#if BCAST_C
        {
            cb_pre_c.wait_front(num_tiles_per_cycle);
            cb_bcast_c.reserve_back(num_tiles_per_cycle);
            unary_bcast_init<BroadcastType::ROW>(cb_pre_c.get_cb_id(), cb_bcast_c.get_cb_id());

            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(cb_pre_c.get_cb_id(), 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_bcast_c.get_cb_id());
            cb_bcast_c.push_back(num_tiles_per_cycle);
            tile_regs_release();

            cb_pre_c.pop_front(num_tiles_per_cycle);
        }
#endif

        // Now execute the ternary SFPU operation on the effective inputs.
        // Reserve output when ready to write.
        cb_out.reserve_back(num_tiles_per_cycle);

        // Ensure fronts are ready for whichever CBs we will read from
        cb_eff_a.wait_front(num_tiles_per_cycle);
        cb_eff_b.wait_front(num_tiles_per_cycle);
        cb_eff_c.wait_front(num_tiles_per_cycle);

        tile_regs_acquire();

        // Load A -> DST[0], B -> DST[1], C -> DST[2]
        copy_tile_to_dst_init_short(cb_eff_a.get_cb_id());
        copy_tile(cb_eff_a.get_cb_id(), 0, 0);

        copy_tile_to_dst_init_short(cb_eff_b.get_cb_id());
        copy_tile(cb_eff_b.get_cb_id(), 0, 1);

        copy_tile_to_dst_init_short(cb_eff_c.get_cb_id());
        copy_tile(cb_eff_c.get_cb_id(), 0, 2);

        // Execute configured ternary SFPU op
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        tile_regs_commit();
        tile_regs_wait();

        // Write out result
        pack_tile(0, cb_out.get_cb_id());
        tile_regs_release();

        cb_out.push_back(num_tiles_per_cycle);

        // Pop fronts for the consumed inputs.
        cb_eff_a.pop_front(num_tiles_per_cycle);
        cb_eff_b.pop_front(num_tiles_per_cycle);
        cb_eff_c.pop_front(num_tiles_per_cycle);
    }
}
