// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/where.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // typically 1

    // Pre-CBs for inputs A, B, C and output CB
    constexpr auto cb_pre_a = tt::CBIndex::c_0;
    constexpr auto cb_pre_b = tt::CBIndex::c_1;
    constexpr auto cb_pre_c = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    // CBs to hold LLK broadcast results when enabled
    constexpr auto cb_bcast_a = tt::CBIndex::c_4;
    constexpr auto cb_bcast_b = tt::CBIndex::c_5;
    constexpr auto cb_bcast_c = tt::CBIndex::c_6;

// Compile-time effective CB selection
#if BCAST_A
    constexpr auto cb_eff_a = cb_bcast_a;
#else
    constexpr auto cb_eff_a = cb_pre_a;
#endif
#if BCAST_B
    constexpr auto cb_eff_b = cb_bcast_b;
#else
    constexpr auto cb_eff_b = cb_pre_b;
#endif
#if BCAST_C
    constexpr auto cb_eff_c = cb_bcast_c;
#else
    constexpr auto cb_eff_c = cb_pre_c;
#endif

    // Initialize pack/format for SFPU-style ternary kernels (matches existing ternary SFPU kernels)
    unary_op_init_common(cb_eff_a, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
#if BCAST_A
        {
            cb_wait_front(cb_pre_a, num_tiles_per_cycle);
            cb_reserve_back(cb_bcast_a, num_tiles_per_cycle);
            unary_bcast_init<BroadcastType::ROW>(cb_pre_a, cb_bcast_a);

            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(cb_pre_a, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_bcast_a);
            cb_push_back(cb_bcast_a, num_tiles_per_cycle);
            tile_regs_release();

            cb_pop_front(cb_pre_a, num_tiles_per_cycle);
        }
#endif

#if BCAST_B
        {
            cb_wait_front(cb_pre_b, num_tiles_per_cycle);
            cb_reserve_back(cb_bcast_b, num_tiles_per_cycle);
            unary_bcast_init<BroadcastType::ROW>(cb_pre_b, cb_bcast_b);

            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(cb_pre_b, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_bcast_b);
            cb_push_back(cb_bcast_b, num_tiles_per_cycle);
            tile_regs_release();

            cb_pop_front(cb_pre_b, num_tiles_per_cycle);
        }
#endif

#if BCAST_C
        {
            cb_wait_front(cb_pre_c, num_tiles_per_cycle);
            cb_reserve_back(cb_bcast_c, num_tiles_per_cycle);
            unary_bcast_init<BroadcastType::ROW>(cb_pre_c, cb_bcast_c);

            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(cb_pre_c, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_bcast_c);
            cb_push_back(cb_bcast_c, num_tiles_per_cycle);
            tile_regs_release();

            cb_pop_front(cb_pre_c, num_tiles_per_cycle);
        }
#endif

        // Now execute the ternary SFPU operation on the effective inputs.
        // Reserve output when ready to write.
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // Ensure fronts are ready for whichever CBs we will read from
        cb_wait_front(cb_eff_a, num_tiles_per_cycle);
        cb_wait_front(cb_eff_b, num_tiles_per_cycle);
        cb_wait_front(cb_eff_c, num_tiles_per_cycle);

        tile_regs_acquire();

        // Load A -> DST[0], B -> DST[1], C -> DST[2]
        copy_tile_to_dst_init_short(cb_eff_a);
        copy_tile(cb_eff_a, 0, 0);

        copy_tile_to_dst_init_short(cb_eff_b);
        copy_tile(cb_eff_b, 0, 1);

        copy_tile_to_dst_init_short(cb_eff_c);
        copy_tile(cb_eff_c, 0, 2);

        // Execute configured ternary SFPU op
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        tile_regs_commit();
        tile_regs_wait();

        // Write out result
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);

        // Pop fronts for the consumed inputs.
        cb_pop_front(cb_eff_a, num_tiles_per_cycle);
        cb_pop_front(cb_eff_b, num_tiles_per_cycle);
        cb_pop_front(cb_eff_c, num_tiles_per_cycle);
    }
}
}  // namespace NAMESPACE
