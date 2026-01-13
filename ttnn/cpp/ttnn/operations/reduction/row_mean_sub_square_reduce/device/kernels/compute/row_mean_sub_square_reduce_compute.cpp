// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api.h"  // for square_tile, square_tile_init

// Helper library includes
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

/**
 * Compute kernel for row_mean_sub_square_reduce operation
 *
 * Implements the 5-phase pipeline per design document:
 * - Phase 1: Tilize (USE HELPER)
 * - Phase 2: Reduce mean with PERSISTENT mode (USE HELPER)
 * - Phase 3: Subtract mean + square (NO HELPER - raw calls)
 * - Phase 4: Reduce variance (USE HELPER)
 * - Phase 5: Untilize (USE HELPER)
 *
 * CRITICAL: Do NOT add CB operations around helper calls - helpers are self-contained.
 */

namespace NAMESPACE {

void MAIN {
    // Compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);                 // Tiles per row
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(1);  // Tile-rows to process

    // CB indices from spec
    constexpr uint32_t cb_rm_in = tt::CBIndex::c_0;         // Row-major input
    constexpr uint32_t cb_tilized = tt::CBIndex::c_1;       // Tilized data
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;        // Scaler (1/W)
    constexpr uint32_t cb_mean = tt::CBIndex::c_3;          // Mean tile
    constexpr uint32_t cb_intermediate = tt::CBIndex::c_4;  // Squared differences
    constexpr uint32_t cb_out_tiled = tt::CBIndex::c_5;     // Variance (tiled)
    constexpr uint32_t cb_rm_out = tt::CBIndex::c_16;       // Row-major output

    // Hardware initialization (REQUIRED before any helper)
    // Takes (input_cb, output_cb) - use first input CB and first output CB
    compute_kernel_hw_startup(cb_rm_in, cb_tilized);

    // Process each tile-row
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // =====================================================================
        // Phase 1: Tilize (USE HELPER)
        // Converts 32 row-major sticks to Wt tiles
        // Helper handles: cb_wait, cb_reserve, tilize_block, cb_push, cb_pop
        // =====================================================================
        compute_kernel_lib::tilize(cb_rm_in, Wt, cb_tilized, 1);

        // =====================================================================
        // Phase 2: Reduce Mean with PERSISTENT mode (USE HELPER)
        // Reduces Wt tiles along W to produce mean tile
        // PERSISTENT mode keeps tiles in cb_tilized for Phase 3 (NO pop)
        // Helper handles: tile_regs, reduce_init, reduce_tile, pack_tile
        // =====================================================================
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputMode::PERSISTENT>(
                cb_tilized, cb_scaler, cb_mean, compute_kernel_lib::TileShape::row(Wt));

        // =====================================================================
        // Phase 3: Subtract Mean + Square (NO HELPER - raw calls)
        // For each input tile: diff = x - mean, sq = diff^2
        // Input: cb_tilized (Wt tiles still present from PERSISTENT reduce)
        // Input: cb_mean (1 tile with mean values in column 0, one per row)
        // Output: cb_intermediate (Wt tiles)
        //
        // Uses:
        // - sub_bcast_cols: dst[0] = tilized[wt] - mean (broadcast column 0 to all columns)
        //   This correctly subtracts each row's mean from that row's values
        // - square_tile: dst[0] = dst[0]^2 (SFPU square in-place)
        // =====================================================================

        // Init subtraction operation with column broadcast (requires both CB IDs)
        // BroadcastType::COL broadcasts column 0 of mean tile to all columns
        sub_bcast_cols_init_short(cb_tilized, cb_mean);

        // Wait for mean tile (produced by Phase 2)
        cb_wait_front(cb_mean, 1);

        // Process each input tile
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            // Input tile already in cb_tilized (PERSISTENT mode kept it)
            // Access via index since we didn't pop

            tile_regs_acquire();

            // Subtract: dst[0] = tilized[wt] - mean (broadcast column)
            // sub_tiles_bcast_cols broadcasts column 0 of mean tile to all columns
            // This correctly handles row-wise mean subtraction
            sub_tiles_bcast_cols(cb_tilized, cb_mean, wt, 0, 0);

            // Square: dst[0] = dst[0]^2 using SFPU square (in-place on DST)
            square_tile_init();
            square_tile(0);

            // Pack the squared difference to cb_intermediate
            cb_reserve_back(cb_intermediate, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_intermediate);
            tile_regs_release();
            cb_push_back(cb_intermediate, 1);
        }

        // Pop the tiles from cb_tilized (Wt tiles) and cb_mean (1 tile)
        cb_pop_front(cb_tilized, Wt);
        cb_pop_front(cb_mean, 1);

        // =====================================================================
        // Phase 4: Reduce Variance (USE HELPER)
        // Reduces Wt squared difference tiles to variance tile
        // Helper handles: tile_regs, reduce_init, reduce_tile, pack_tile
        // =====================================================================
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_intermediate, cb_scaler, cb_out_tiled, compute_kernel_lib::TileShape::row(Wt));

        // =====================================================================
        // Phase 5: Untilize (USE HELPER)
        // Converts variance tile to row-major format
        // Helper handles: cb_wait, cb_reserve, untilize/pack_untilize, cb_push, cb_pop
        // Template params: tile_width=1, icb=cb_out_tiled, ocb=cb_rm_out
        // =====================================================================
        compute_kernel_lib::untilize<1, cb_out_tiled, cb_rm_out>(1);
    }
}

}  // namespace NAMESPACE
