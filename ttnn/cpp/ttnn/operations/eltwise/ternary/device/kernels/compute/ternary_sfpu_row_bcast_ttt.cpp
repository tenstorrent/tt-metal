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
    constexpr auto cb_pre_a_id = tt::CBIndex::c_0;
    constexpr auto cb_pre_b_id = tt::CBIndex::c_1;
    constexpr auto cb_pre_c_id = tt::CBIndex::c_2;
    constexpr auto cb_out_id = tt::CBIndex::c_3;

    // CBs to hold LLK broadcast results when enabled
    constexpr auto cb_bcast_a_id = tt::CBIndex::c_4;
    constexpr auto cb_bcast_b_id = tt::CBIndex::c_5;
    constexpr auto cb_bcast_c_id = tt::CBIndex::c_6;

// Compile-time effective CB selection
#if BCAST_A
    constexpr auto cb_eff_a_id = cb_bcast_a_id;
#else
    constexpr auto cb_eff_a_id = cb_pre_a_id;
#endif
#if BCAST_B
    constexpr auto cb_eff_b_id = cb_bcast_b_id;
#else
    constexpr auto cb_eff_b_id = cb_pre_b_id;
#endif
#if BCAST_C
    constexpr auto cb_eff_c_id = cb_bcast_c_id;
#else
    constexpr auto cb_eff_c_id = cb_pre_c_id;
#endif

    CircularBuffer cb_out(cb_out_id);
#if BCAST_A
    CircularBuffer cb_pre_a(cb_pre_a_id);
    CircularBuffer cb_bcast_a(cb_bcast_a_id);
#endif
#if BCAST_B
    CircularBuffer cb_pre_b(cb_pre_b_id);
    CircularBuffer cb_bcast_b(cb_bcast_b_id);
#endif
#if BCAST_C
    CircularBuffer cb_pre_c(cb_pre_c_id);
    CircularBuffer cb_bcast_c(cb_bcast_c_id);
#endif
    CircularBuffer cb_eff_a(cb_eff_a_id);
    CircularBuffer cb_eff_b(cb_eff_b_id);
    CircularBuffer cb_eff_c(cb_eff_c_id);

    // Initialize pack/format for SFPU-style ternary kernels (matches existing ternary SFPU kernels)
    unary_op_init_common(cb_eff_a_id, cb_out_id);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
#if BCAST_A
        {
            cb_pre_a.wait_front(num_tiles_per_cycle);
            cb_bcast_a.reserve_back(num_tiles_per_cycle);
            unary_bcast_init<BroadcastType::ROW>(cb_pre_a_id, cb_bcast_a_id);

            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(cb_pre_a_id, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_bcast_a_id);
            cb_bcast_a.push_back(num_tiles_per_cycle);
            tile_regs_release();

            cb_pre_a.pop_front(num_tiles_per_cycle);
        }
#endif

#if BCAST_B
        {
            cb_pre_b.wait_front(num_tiles_per_cycle);
            cb_bcast_b.reserve_back(num_tiles_per_cycle);
            unary_bcast_init<BroadcastType::ROW>(cb_pre_b_id, cb_bcast_b_id);

            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(cb_pre_b_id, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_bcast_b_id);
            cb_bcast_b.push_back(num_tiles_per_cycle);
            tile_regs_release();

            cb_pre_b.pop_front(num_tiles_per_cycle);
        }
#endif

#if BCAST_C
        {
            cb_pre_c.wait_front(num_tiles_per_cycle);
            cb_bcast_c.reserve_back(num_tiles_per_cycle);
            unary_bcast_init<BroadcastType::ROW>(cb_pre_c_id, cb_bcast_c_id);

            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(cb_pre_c_id, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_bcast_c_id);
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
        copy_tile_to_dst_init_short(cb_eff_a_id);
        copy_tile(cb_eff_a_id, 0, 0);

        copy_tile_to_dst_init_short(cb_eff_b_id);
        copy_tile(cb_eff_b_id, 0, 1);

        copy_tile_to_dst_init_short(cb_eff_c_id);
        copy_tile(cb_eff_c_id, 0, 2);

        // Execute configured ternary SFPU op
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        tile_regs_commit();
        tile_regs_wait();

        // Write out result
        pack_tile(0, cb_out_id);
        tile_regs_release();

        cb_out.push_back(num_tiles_per_cycle);

        // Pop fronts for the consumed inputs.
        cb_eff_a.pop_front(num_tiles_per_cycle);
        cb_eff_b.pop_front(num_tiles_per_cycle);
        cb_eff_c.pop_front(num_tiles_per_cycle);
    }
}
