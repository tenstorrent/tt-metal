// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace NAMESPACE {

void MAIN {
    // Compile-time args
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // CB IDs
    constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;          // Input RM sticks
    constexpr uint32_t cb_in_tiled = tt::CBIndex::c_1;       // Tiled input
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;         // Scaler tile (1/W)
    constexpr uint32_t cb_reduced_tiled = tt::CBIndex::c_3;  // Reduced tiled output
    constexpr uint32_t cb_out_rm = tt::CBIndex::c_16;        // Output RM sticks

    // Initialize compute kernel hardware
    // Provide representative CBs for initialization (input RM, scaler, output RM)
    compute_kernel_hw_startup(cb_in_rm, cb_scaler, cb_out_rm);

    // Process each tile-row
    for (uint32_t block = 0; block < Ht; ++block) {
        // Phase 1: Tilize one tile-row (32 sticks -> Wt tiles)
        // Helper handles: cb_wait_front(cb_in_rm), cb_reserve_back(cb_in_tiled),
        //                 tilize_block, cb_push_back(cb_in_tiled), cb_pop_front(cb_in_rm)
        compute_kernel_lib::tilize(cb_in_rm, Wt, cb_in_tiled, 1);

        // Phase 2: Reduce the tile-row along width dimension
        // Using SUM with pre-computed 1/W scaler (equivalent to mean)
        // Helper handles: all CB operations and DST management internally
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_in_tiled, cb_scaler, cb_reduced_tiled, compute_kernel_lib::TileShape::row(Wt));

        // Phase 3: Untilize the reduced result (1 tile -> 32 output sticks)
        // Helper handles: cb_wait_front(cb_reduced_tiled), cb_reserve_back(cb_out_rm),
        //                 pack_untilize, cb_push_back(cb_out_rm), cb_pop_front(cb_reduced_tiled)
        compute_kernel_lib::untilize<1, cb_reduced_tiled, cb_out_rm>(1);
    }
}

}  // namespace NAMESPACE
