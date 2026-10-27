// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Expects reader kernel to not perform per-tile fill (FILL_TILE_WITH_FIRST_ROW) controlled by BCAST_LLK flag

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t scalar_arg = get_arg(args::scalar_arg);

    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);  // set to 1
    const bool scalar_is_not_1 = scalar_arg != 1u;

    DataflowBuffer dfb_in0(dfb::in_a);  // input_a
    DataflowBuffer dfb_in1(dfb::in_b);  // input_b
    DataflowBuffer dfb_in2(dfb::in_c);  // input_c
    DataflowBuffer dfb_out(dfb::out);

    // LLK broadcast destination DFBs (dedicated per-input). Each one is bound only when its input is broadcast.
#if BCAST_A
    DataflowBuffer dfb_llk_a(dfb::bcast_a);  // broadcasted A (if used)
#endif
#if BCAST_B
    DataflowBuffer dfb_llk_b(dfb::bcast_b);  // broadcasted B (if used)
#endif
#if BCAST_C
    DataflowBuffer dfb_llk_c(dfb::bcast_c);  // broadcasted C (if used)
#endif

// Effective DFBs chosen at compile-time depending on broadcast flags.
// eff_a, eff_b and eff_c name the DFB the op reads each input from: its row-broadcast staging DFB when that input
// is broadcast, and the input DFB itself otherwise.
#if BCAST_A
    auto& dfb_eff_a = dfb_llk_a;
    constexpr auto eff_a = dfb::bcast_a;
#else
    auto& dfb_eff_a = dfb_in0;
    constexpr auto eff_a = dfb::in_a;
#endif
#if BCAST_B
    auto& dfb_eff_b = dfb_llk_b;
    constexpr auto eff_b = dfb::bcast_b;
#else
    auto& dfb_eff_b = dfb_in1;
    constexpr auto eff_b = dfb::in_b;
#endif
#if BCAST_C
    auto& dfb_eff_c = dfb_llk_c;
    constexpr auto eff_c = dfb::bcast_c;
#else
    auto& dfb_eff_c = dfb_in2;
    constexpr auto eff_c = dfb::in_c;
#endif

    // Initialize binary unit for B*C path; output packer initialized with dfb_out
    compute_kernel_hw_startup(eff_b, eff_c, dfb::out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
// 1) Prepare B and C (broadcast if required), then compute mul(B, C) -> DST[0]
// Perform LLK row broadcast for B if requested
#if BCAST_B
        dfb_in1.wait_front(num_tiles_per_cycle);
        dfb_llk_b.reserve_back(num_tiles_per_cycle);
        reconfig_data_format(dfb::in_b, dfb::in_b);
        pack_reconfig_data_format(dfb::bcast_b);
        unary_bcast_init<BroadcastType::ROW>(dfb::in_b);
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(dfb::in_b, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::bcast_b);
        dfb_llk_b.push_back(num_tiles_per_cycle);
        tile_regs_release();
        dfb_in1.pop_front(num_tiles_per_cycle);
#endif

// Perform LLK row broadcast for C if requested
#if BCAST_C
        dfb_in2.wait_front(num_tiles_per_cycle);
        dfb_llk_c.reserve_back(num_tiles_per_cycle);
        reconfig_data_format(dfb::in_c, dfb::in_c);
        pack_reconfig_data_format(dfb::bcast_c);
        unary_bcast_init<BroadcastType::ROW>(dfb::in_c);
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(dfb::in_c, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::bcast_c);
        dfb_llk_c.push_back(num_tiles_per_cycle);
        tile_regs_release();
        dfb_in2.pop_front(num_tiles_per_cycle);
#endif

// Perform LLK row broadcast for A if requested (do this before compute regs session)
#if BCAST_A
        dfb_in0.wait_front(num_tiles_per_cycle);
        dfb_llk_a.reserve_back(num_tiles_per_cycle);
        reconfig_data_format(dfb::in_a, dfb::in_a);
        pack_reconfig_data_format(dfb::bcast_a);
        unary_bcast_init<BroadcastType::ROW>(dfb::in_a);
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(dfb::in_a, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::bcast_a);
        dfb_llk_a.push_back(num_tiles_per_cycle);
        tile_regs_release();
        dfb_in0.pop_front(num_tiles_per_cycle);
#endif

        // Ensure sources available
        dfb_eff_a.wait_front(num_tiles_per_cycle);
        dfb_eff_b.wait_front(num_tiles_per_cycle);
        dfb_eff_c.wait_front(num_tiles_per_cycle);

        tile_regs_acquire();

        copy_init(eff_a);
        copy_tile(eff_a, 0 /*in_tile_index*/, 0 /*dst_tile_index*/);

        copy_init(eff_b);
        copy_tile(eff_b, 0 /*in_tile_index*/, 1 /*dst_tile_index*/);

        copy_init(eff_c);
        copy_tile(eff_c, 0 /*in_tile_index*/, 2 /*dst_tile_index*/);

        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg);

        tile_regs_commit();
        tile_regs_wait();

        // Reserve output buffer only when ready to write
        dfb_out.reserve_back(num_tiles_per_cycle);

        // Pack the result from DST[0] to output
        pack_tile(0, dfb::out);

        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);
        dfb_eff_a.pop_front(num_tiles_per_cycle);
        dfb_eff_b.pop_front(num_tiles_per_cycle);
        dfb_eff_c.pop_front(num_tiles_per_cycle);
    }
}
