// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/cb_policies.hpp"

namespace NAMESPACE {

void MAIN {
    // ============================================================
    // Compile-time args
    // ============================================================
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    // asd
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // ============================================================
    // CB IDs (matching kernel_design.md)
    // ============================================================
    constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;               // Input RM sticks
    constexpr uint32_t cb_in_tiled = tt::CBIndex::c_1;            // Tiled input (PERSISTENT phases 1-3)
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;              // Scaler tile (1/W)
    constexpr uint32_t cb_mean_tiled = tt::CBIndex::c_3;          // Mean tile
    constexpr uint32_t cb_centralized_tiled = tt::CBIndex::c_4;   // Centralized tiles (PERSISTENT phases 3-8)
    constexpr uint32_t cb_squared_tiled = tt::CBIndex::c_5;       // Squared tiles
    constexpr uint32_t cb_variance_tiled = tt::CBIndex::c_6;      // Variance tile
    constexpr uint32_t cb_epsilon = tt::CBIndex::c_7;             // Epsilon scalar tile
    constexpr uint32_t cb_rsqrt_tiled = tt::CBIndex::c_8;         // Rsqrt result tile
    constexpr uint32_t cb_standardized_tiled = tt::CBIndex::c_9;  // Standardized/output tiled (reused)
    constexpr uint32_t cb_gamma_rm = tt::CBIndex::c_10;           // Gamma RM sticks
    constexpr uint32_t cb_gamma_tiled = tt::CBIndex::c_11;        // Gamma tiled (PROGRAM LIFETIME)
    constexpr uint32_t cb_beta_rm = tt::CBIndex::c_12;            // Beta RM sticks
    constexpr uint32_t cb_beta_tiled = tt::CBIndex::c_13;         // Beta tiled (PROGRAM LIFETIME)
    constexpr uint32_t cb_scaled_tiled = tt::CBIndex::c_14;       // Scaled tiles (Phase 9 gamma multiply output)
    constexpr uint32_t cb_out_rm = tt::CBIndex::c_16;             // Output RM sticks (Phase 11 untilize output)

    // ============================================================
    // Initialize compute kernel hardware
    // ============================================================
    compute_kernel_hw_startup(cb_in_rm, cb_scaler, cb_out_rm);

    // ============================================================
    // Custom CB policies (per kernel_design.md)
    // ============================================================
    // PreloadedPopAtEnd: tiles already present in CB (from PERSISTENT reduce), pop all at end
    using PreloadedPopAtEnd = cb_policies::InputPolicy<cb_policies::WaitCallerManaged, cb_policies::PopAtEnd>;
    // PreloadedNoPop: tiles already present in CB, do NOT pop (needed for later phase or program lifetime)
    using PreloadedNoPop = cb_policies::InputPolicy<cb_policies::WaitCallerManaged, cb_policies::PopNever>;
    // WaitUpfrontPopAtEnd: wait for tiles upfront, pop all at end
    using WaitUpfrontPopAtEnd = cb_policies::InputPolicy<cb_policies::WaitUpfront, cb_policies::PopAtEnd>;

    // ============================================================
    // PRE-LOOP: Tilize Gamma and Beta (ONCE at program start)
    // These persist for the entire program lifetime
    // ============================================================

    // Tilize gamma (cb_gamma_rm -> cb_gamma_tiled)
    // USE HELPER: compute_kernel_lib::tilize()
    // Helper handles: cb_wait_front(cb_gamma_rm), cb_reserve_back(cb_gamma_tiled),
    //                 tilize_block, cb_push_back(cb_gamma_tiled), cb_pop_front(cb_gamma_rm)
    compute_kernel_lib::tilize(cb_gamma_rm, Wt, cb_gamma_tiled, 1);

    // Tilize beta (cb_beta_rm -> cb_beta_tiled)
    // USE HELPER: compute_kernel_lib::tilize()
    compute_kernel_lib::tilize(cb_beta_rm, Wt, cb_beta_tiled, 1);

    // Wait for gamma and beta tiles ONCE before the loop.
    // These tiles persist for the entire program lifetime and are never popped.
    // The PreloadedNoPop policy expects WaitCallerManaged, so we wait here.
    cb_wait_front(cb_gamma_tiled, Wt);
    cb_wait_front(cb_beta_tiled, Wt);

    // ============================================================
    // Process each tile-row
    // ============================================================
    for (uint32_t block = 0; block < Ht; ++block) {
        // ========================================================
        // Phase 1: Tilize (RM sticks -> tiled)
        // USE HELPER: compute_kernel_lib::tilize()
        // Helper handles: cb_wait_front(cb_in_rm), cb_reserve_back(cb_in_tiled),
        //                 tilize_block, cb_push_back(cb_in_tiled), cb_pop_front(cb_in_rm)
        // ========================================================
        compute_kernel_lib::tilize(cb_in_rm, Wt, cb_in_tiled, 1);

        // ========================================================
        // Phase 2: Reduce (Mean) with PERSISTENT mode
        // USE HELPER: compute_kernel_lib::reduce<SUM, REDUCE_ROW, PERSISTENT>()
        // PERSISTENT mode: waits for all Wt tiles but does NOT pop them
        // This keeps tiles in cb_in_tiled for Phase 3
        // ========================================================
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputMode::PERSISTENT>(
                cb_in_tiled, cb_scaler, cb_mean_tiled, compute_kernel_lib::TileShape::row(Wt));

        // ========================================================
        // Phase 3: Broadcast Subtract (Centralize)
        // USE HELPER: compute_kernel_lib::sub<COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>()
        // cb_in_tiled: original tiled data (Wt tiles, still in CB from PERSISTENT reduce)
        // cb_mean_tiled: mean tile (1 tile, Col0 valid from REDUCE_ROW)
        // Output to cb_centralized_tiled (Wt tiles)
        // BroadcastDim::COL because REDUCE_ROW produces column-shaped output
        // ========================================================
        compute_kernel_lib::sub<compute_kernel_lib::BroadcastDim::COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>(
            cb_in_tiled, cb_mean_tiled, cb_centralized_tiled, compute_kernel_lib::BinaryTileShape::row(Wt));

        // ========================================================
        // Phase 4: Square (element-wise self-multiply)
        // USE HELPER: compute_kernel_lib::binary_op<SQUARE, NONE, PreloadedNoPop>()
        // CRITICAL: PreloadedNoPop ensures cb_centralized_tiled tiles persist for Phase 8
        // cb_centralized_tiled: centralized data (Wt tiles, just pushed by Phase 3)
        // Output to cb_squared_tiled (Wt tiles)
        // ========================================================
        // Wait upfront for the centralized tiles from phase 3
        cb_wait_front(cb_centralized_tiled, Wt);
        compute_kernel_lib::
            binary_op<compute_kernel_lib::BinaryOpType::SQUARE, compute_kernel_lib::BroadcastDim::NONE, PreloadedNoPop>(
                cb_centralized_tiled,
                cb_centralized_tiled,
                cb_squared_tiled,
                compute_kernel_lib::BinaryTileShape::row(Wt));

        // ========================================================
        // Phase 5: Reduce (Variance) with STREAMING mode
        // USE HELPER: compute_kernel_lib::reduce<SUM, REDUCE_ROW, STREAMING>()
        // STREAMING mode: processes and pops tiles one at a time
        // ========================================================
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputMode::STREAMING>(
                cb_squared_tiled, cb_scaler, cb_variance_tiled, compute_kernel_lib::TileShape::row(Wt));

        // ========================================================
        // Phases 6-7: Add Epsilon + Rsqrt (Combined, NO HELPER)
        // Compute rsqrt(variance + epsilon) using DST registers directly
        // This is a specialized pattern; intermediate stays in DST
        // ========================================================
        {
            // Wait for inputs
            cb_wait_front(cb_variance_tiled, 1);  // variance
            cb_wait_front(cb_epsilon, 1);         // epsilon (pushed once by reader, never popped)
            cb_reserve_back(cb_rsqrt_tiled, 1);   // rsqrt output

            tile_regs_acquire();

            // Copy variance to DST[0]
            copy_tile_to_dst_init_short_with_dt(cb_epsilon, cb_variance_tiled);
            copy_tile(cb_variance_tiled, 0, 0);  // cb_variance_tiled tile 0 -> DST[0]

            // Add epsilon from cb_epsilon
            // Note: add_binary_tile_init does not take arguments
            add_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(cb_variance_tiled, cb_epsilon);
            copy_tile(cb_epsilon, 0, 1);  // cb_epsilon tile 0 -> DST[1]
            add_binary_tile(0, 1, 0);     // DST[0] = DST[0] + DST[1]

            // Rsqrt
            rsqrt_tile_init();
            rsqrt_tile(0);  // DST[0] = rsqrt(DST[0])

            // Pack result
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_rsqrt_tiled);
            tile_regs_release();

            cb_push_back(cb_rsqrt_tiled, 1);
            cb_pop_front(cb_variance_tiled, 1);
            // Note: cb_epsilon NOT popped (program lifetime)
        }

        // ========================================================
        // Phase 8: Broadcast Multiply (Standardize)
        // USE HELPER: compute_kernel_lib::mul<COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>()
        // cb_centralized_tiled: centralized tiles (Wt tiles, still present from Phase 4 NO POP)
        // cb_rsqrt_tiled: rsqrt tile (1 tile, Col0 valid)
        // Output to cb_standardized_tiled (Wt tiles)
        // BroadcastDim::COL to replicate rsqrt across columns
        // ========================================================
        compute_kernel_lib::mul<compute_kernel_lib::BroadcastDim::COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>(
            cb_centralized_tiled, cb_rsqrt_tiled, cb_standardized_tiled, compute_kernel_lib::BinaryTileShape::row(Wt));

        // ========================================================
        // Phase 9: Broadcast Multiply Gamma
        // USE HELPER: compute_kernel_lib::mul<ROW, WaitUpfrontPopAtEnd, PreloadedNoPop>()
        // cb_standardized_tiled: standardized tiles (Wt tiles)
        // cb_gamma_tiled: gamma tiles (Wt tiles, Row0 valid - program lifetime)
        // Output to cb_scaled_tiled (Wt tiles)
        // BroadcastDim::ROW because gamma is [1, Wt] shape, replicate down
        // NOTE: Using WaitUpfrontPopAtEnd instead of Streaming to ensure all A tiles
        // are available before processing, which allows correct B tile indexing
        // ========================================================
        compute_kernel_lib::mul<compute_kernel_lib::BroadcastDim::ROW, WaitUpfrontPopAtEnd, PreloadedNoPop>(
            cb_standardized_tiled, cb_gamma_tiled, cb_scaled_tiled, compute_kernel_lib::BinaryTileShape::row(Wt));

        // ========================================================
        // Phase 10: Broadcast Add Beta
        // USE HELPER: compute_kernel_lib::add<ROW, WaitUpfrontPopAtEnd, PreloadedNoPop>()
        // cb_scaled_tiled: scaled tiles (Wt tiles)
        // cb_beta_tiled: beta tiles (Wt tiles, Row0 valid - program lifetime)
        // Output to cb_standardized_tiled (Wt tiles) - REUSED CB
        // BroadcastDim::ROW because beta is [1, Wt] shape, replicate down
        // ========================================================
        compute_kernel_lib::add<compute_kernel_lib::BroadcastDim::ROW, WaitUpfrontPopAtEnd, PreloadedNoPop>(
            cb_scaled_tiled, cb_beta_tiled, cb_standardized_tiled, compute_kernel_lib::BinaryTileShape::row(Wt));

        // ========================================================
        // Phase 11: Untilize (tiled -> RM sticks)
        // USE HELPER: compute_kernel_lib::untilize<Wt, cb_standardized_tiled, cb_out_rm>()
        // Input: cb_standardized_tiled (c_9) - tiled output after beta add
        // Output: cb_out_rm (c_16) - RM sticks for writer
        // CRITICAL: These must be DIFFERENT CBs - untilize reads tiles and writes sticks
        // Helper handles: cb_wait_front, pack_untilize, cb_push_back, cb_pop_front
        // ========================================================
        compute_kernel_lib::untilize<Wt, cb_standardized_tiled, cb_out_rm>(1);
    }
}

}  // namespace NAMESPACE
