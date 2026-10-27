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
#include "api/compute/eltwise_unary/mac.h"
#include "api/compute/eltwise_unary/snake_beta.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);  // typically 1

    // Pre-DFBs for inputs A, B, C and output DFB
    DataflowBuffer dfb_out(dfb::out);
    // eff_a, eff_b and eff_c name the DFB the op reads each input from: its row-broadcast staging DFB when that
    // input is broadcast, and the input DFB itself otherwise.
#if BCAST_A
    DataflowBuffer dfb_pre_a(dfb::in_a);
    DataflowBuffer dfb_bcast_a(dfb::bcast_a);
    DataflowBuffer dfb_eff_a = dfb_bcast_a;
    constexpr auto eff_a = dfb::bcast_a;
#else
    DataflowBuffer dfb_eff_a(dfb::in_a);
    constexpr auto eff_a = dfb::in_a;
#endif
#if BCAST_B
    DataflowBuffer dfb_pre_b(dfb::in_b);
    DataflowBuffer dfb_bcast_b(dfb::bcast_b);
    DataflowBuffer dfb_eff_b = dfb_bcast_b;
    constexpr auto eff_b = dfb::bcast_b;
#else
    DataflowBuffer dfb_eff_b(dfb::in_b);
    constexpr auto eff_b = dfb::in_b;
#endif
#if BCAST_C
    DataflowBuffer dfb_pre_c(dfb::in_c);
    DataflowBuffer dfb_bcast_c(dfb::bcast_c);
    DataflowBuffer dfb_eff_c = dfb_bcast_c;
    constexpr auto eff_c = dfb::bcast_c;
#else
    DataflowBuffer dfb_eff_c(dfb::in_c);
    constexpr auto eff_c = dfb::in_c;
#endif

    // Initialize pack/format for SFPU-style ternary kernels (matches existing ternary SFPU kernels)
    compute_kernel_hw_startup(eff_a, dfb::out);
    copy_init(eff_a);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
#if BCAST_A
        {
            dfb_pre_a.wait_front(num_tiles_per_cycle);
            dfb_bcast_a.reserve_back(num_tiles_per_cycle);
            reconfig_data_format(dfb::in_a, dfb::in_a);
            pack_reconfig_data_format(dfb::bcast_a);
            unary_bcast_init<BroadcastType::ROW>(dfb::in_a);

            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(dfb::in_a, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, dfb::bcast_a);
            dfb_bcast_a.push_back(num_tiles_per_cycle);
            tile_regs_release();

            dfb_pre_a.pop_front(num_tiles_per_cycle);
        }
#endif

#if BCAST_B
        {
            dfb_pre_b.wait_front(num_tiles_per_cycle);
            dfb_bcast_b.reserve_back(num_tiles_per_cycle);
            reconfig_data_format(dfb::in_b, dfb::in_b);
            pack_reconfig_data_format(dfb::bcast_b);
            unary_bcast_init<BroadcastType::ROW>(dfb::in_b);

            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(dfb::in_b, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, dfb::bcast_b);
            dfb_bcast_b.push_back(num_tiles_per_cycle);
            tile_regs_release();

            dfb_pre_b.pop_front(num_tiles_per_cycle);
        }
#endif

#if BCAST_C
        {
            dfb_pre_c.wait_front(num_tiles_per_cycle);
            dfb_bcast_c.reserve_back(num_tiles_per_cycle);
            reconfig_data_format(dfb::in_c, dfb::in_c);
            pack_reconfig_data_format(dfb::bcast_c);
            unary_bcast_init<BroadcastType::ROW>(dfb::in_c);

            tile_regs_acquire();
            unary_bcast<BroadcastType::ROW>(dfb::in_c, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, dfb::bcast_c);
            dfb_bcast_c.push_back(num_tiles_per_cycle);
            tile_regs_release();

            dfb_pre_c.pop_front(num_tiles_per_cycle);
        }
#endif

        // Now execute the ternary SFPU operation on the effective inputs.
        // Reserve output when ready to write.
        dfb_out.reserve_back(num_tiles_per_cycle);

        // Ensure fronts are ready for whichever DFBs we will read from
        dfb_eff_a.wait_front(num_tiles_per_cycle);
        dfb_eff_b.wait_front(num_tiles_per_cycle);
        dfb_eff_c.wait_front(num_tiles_per_cycle);

        tile_regs_acquire();

        // Load A -> DST[0], B -> DST[1], C -> DST[2]
        copy_init(eff_a);
        copy_tile(eff_a, 0, 0);

        copy_init(eff_b);
        copy_tile(eff_b, 0, 1);

        copy_init(eff_c);
        copy_tile(eff_c, 0, 2);

        // Execute configured ternary SFPU op
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        tile_regs_commit();
        tile_regs_wait();

        // Write out result
        pack_tile(0, dfb::out);
        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);

        // Pop fronts for the consumed inputs.
        dfb_eff_a.pop_front(num_tiles_per_cycle);
        dfb_eff_b.pop_front(num_tiles_per_cycle);
        dfb_eff_c.pop_front(num_tiles_per_cycle);
    }
}
