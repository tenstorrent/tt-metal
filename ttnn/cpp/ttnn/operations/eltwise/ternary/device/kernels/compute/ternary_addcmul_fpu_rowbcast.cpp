// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Expects reader kernel to not perform per-tile fill (FILL_TILE_WITH_FIRST_ROW) controlled by BCAST_LLK flag

#include <cstdint>

#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_unary/addcmul.h"
namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1
    const bool scalar_is_not_1 = scalar_arg != 1u;

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out = tt::CBIndex::c_3;

    // LLK broadcast destination CBs (dedicated per-input):
    constexpr auto cb_llk_a = tt::CBIndex::c_4;  // broadcasted A (if used)
    constexpr auto cb_llk_b = tt::CBIndex::c_5;  // broadcasted B (if used)
    constexpr auto cb_llk_c = tt::CBIndex::c_6;  // broadcasted C (if used)

// Effective CBs chosen at compile-time depending on broadcast flags
#if BCAST_A
    constexpr auto cb_eff_a = cb_llk_a;
#else
    constexpr auto cb_eff_a = cb_in0;
#endif
#if BCAST_B
    constexpr auto cb_eff_b = cb_llk_b;
#else
    constexpr auto cb_eff_b = cb_in1;
#endif
#if BCAST_C
    constexpr auto cb_eff_c = cb_llk_c;
#else
    constexpr auto cb_eff_c = cb_in2;
#endif

    // Initialize binary unit for B*C path; output packer initialized with cb_out
    binary_op_init_common(cb_eff_b, cb_eff_c, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
// 1) Prepare B and C (broadcast if required), then compute mul(B, C) -> DST[0]
// Perform LLK row broadcast for B if requested
#if BCAST_B
        cb_wait_front(cb_in1, num_tiles_per_cycle);
        cb_reserve_back(cb_llk_b, num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(cb_in1, cb_llk_b);
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_in1, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_llk_b);
        cb_push_back(cb_llk_b, num_tiles_per_cycle);
        tile_regs_release();
        cb_pop_front(cb_in1, num_tiles_per_cycle);
#endif

// Perform LLK row broadcast for C if requested
#if BCAST_C
        cb_wait_front(cb_in2, num_tiles_per_cycle);
        cb_reserve_back(cb_llk_c, num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(cb_in2, cb_llk_c);
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_in2, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_llk_c);
        cb_push_back(cb_llk_c, num_tiles_per_cycle);
        tile_regs_release();
        cb_pop_front(cb_in2, num_tiles_per_cycle);
#endif

// Perform LLK row broadcast for A if requested (do this before compute regs session)
#if BCAST_A
        cb_wait_front(cb_in0, num_tiles_per_cycle);
        cb_reserve_back(cb_llk_a, num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(cb_in0, cb_llk_a);
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_in0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_llk_a);
        cb_push_back(cb_llk_a, num_tiles_per_cycle);
        tile_regs_release();
        cb_pop_front(cb_in0, num_tiles_per_cycle);
#endif

        // Ensure sources available
        cb_wait_front(cb_eff_a, num_tiles_per_cycle);
        cb_wait_front(cb_eff_b, num_tiles_per_cycle);
        cb_wait_front(cb_eff_c, num_tiles_per_cycle);

        tile_regs_acquire();

        copy_tile_init(cb_eff_a);
        copy_tile(cb_eff_a, 0 /*in_tile_index*/, 0 /*dst_tile_index*/);

        copy_tile_init(cb_eff_b);
        copy_tile(cb_eff_b, 0 /*in_tile_index*/, 1 /*dst_tile_index*/);

        copy_tile_init(cb_eff_c);
        copy_tile(cb_eff_c, 0 /*in_tile_index*/, 2 /*dst_tile_index*/);

        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg);

        tile_regs_commit();
        tile_regs_wait();

        // Reserve output buffer only when ready to write
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // Pack the result from DST[0] to output
        pack_tile(0, cb_out);

        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_eff_a, num_tiles_per_cycle);
        cb_pop_front(cb_eff_b, num_tiles_per_cycle);
        cb_pop_front(cb_eff_c, num_tiles_per_cycle);
    }
}
}  // namespace NAMESPACE
