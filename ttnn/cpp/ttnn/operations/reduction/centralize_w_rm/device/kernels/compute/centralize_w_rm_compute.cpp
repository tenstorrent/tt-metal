// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace NAMESPACE {

void MAIN {
    // Compile-time args
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // CB IDs
    constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;              // Input RM sticks
    constexpr uint32_t cb_in_tiled = tt::CBIndex::c_1;           // Tiled input (persists for bcast_sub)
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;             // Scaler tile (1/W)
    constexpr uint32_t cb_mean_tiled = tt::CBIndex::c_3;         // Reduced mean tile
    constexpr uint32_t cb_centralized_tiled = tt::CBIndex::c_4;  // Centralized tiled data
    constexpr uint32_t cb_out_rm = tt::CBIndex::c_16;            // Output RM sticks

    // Initialize compute kernel hardware
    // Provide representative CBs for initialization (input RM, scaler, output RM)
    compute_kernel_hw_startup(cb_in_rm, cb_scaler, cb_out_rm);

    // Process each tile-row
    for (uint32_t block = 0; block < Ht; ++block) {
        // Phase 1: Tilize one tile-row (32 sticks -> Wt tiles)
        // Helper handles: cb_wait_front(cb_in_rm), cb_reserve_back(cb_in_tiled),
        //                 tilize_block, cb_push_back(cb_in_tiled), cb_pop_front(cb_in_rm)
        compute_kernel_lib::tilize(cb_in_rm, Wt, cb_in_tiled, 1);

        // Phase 2: Reduce the tile-row along width dimension with PERSISTENT mode
        // Using SUM with pre-computed 1/W scaler (equivalent to mean)
        // PERSISTENT mode: waits for all Wt tiles upfront but does NOT pop them
        // This keeps tiles in cb_in_tiled available for the subsequent bcast_sub
        // Helper handles: all CB operations and DST management internally
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputMode::PERSISTENT>(
                cb_in_tiled, cb_scaler, cb_mean_tiled, compute_kernel_lib::TileShape::row(Wt));

        // Phase 3: Broadcast subtract - original minus mean
        // cb_in_tiled: original tiled data (Wt tiles, still in CB from phase 2 due to PERSISTENT)
        // cb_mean_tiled: mean tile (1 tile, Col0 valid from REDUCE_ROW)
        // Output to cb_centralized_tiled (Wt tiles)
        // Use BroadcastDim::COL because REDUCE_ROW produces column-shaped output
        // Input A policy: Custom - tiles already present (no wait), pop all at end
        //   cb_policies::Preloaded has pops_caller_managed=true which means NO pop
        //   We need InputPolicy<WaitCallerManaged, PopAtEnd> to pop Wt tiles at the end
        // Input B policy: For COL broadcast, we need to wait for 1 B tile upfront and pop at end
        //   cb_policies::Streaming would wait/pop per A tile but we only have 1 B tile
        //   We need InputPolicy<WaitUpfront, PopAtEnd> for the B input (mean tile)
        using PreloadedPopAtEnd = cb_policies::InputPolicy<cb_policies::WaitCallerManaged, cb_policies::PopAtEnd>;
        using WaitUpfrontPopAtEnd = cb_policies::InputPolicy<cb_policies::WaitUpfront, cb_policies::PopAtEnd>;
        compute_kernel_lib::sub<compute_kernel_lib::BroadcastDim::COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>(
            cb_in_tiled, cb_mean_tiled, cb_centralized_tiled, compute_kernel_lib::BinaryTileShape::row(Wt));

        // Phase 4: Untilize the centralized result (Wt tiles -> 32 output sticks)
        // Helper handles: cb_wait_front(cb_centralized_tiled), cb_reserve_back(cb_out_rm),
        //                 pack_untilize, cb_push_back(cb_out_rm), cb_pop_front(cb_centralized_tiled)
        compute_kernel_lib::untilize<Wt, cb_centralized_tiled, cb_out_rm>(1);
    }
}

}  // namespace NAMESPACE
