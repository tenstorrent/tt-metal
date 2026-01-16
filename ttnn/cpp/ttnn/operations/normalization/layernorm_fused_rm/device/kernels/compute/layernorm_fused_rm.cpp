// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/compute_kernel_hw_startup.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"

// Include helper libraries
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

// Compute kernel for layernorm_fused_rm
// Full LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
//
// Algorithm:
// 1. Tilize input
// 2. Compute mean = sum(x) / W (reduce row with 1/W scaler)
// 3. Center: centered = x - mean (broadcast col)
// 4. Square: squared = centered^2
// 5. Compute variance = sum(squared) / W
// 6. Add epsilon and rsqrt: inv_std = 1/sqrt(var + eps)
// 7. Normalize: normalized = centered * inv_std
// 8. Scale: scaled = normalized * gamma (broadcast row)
// 9. Shift: output = scaled + beta (broadcast row)
// 10. Untilize output

namespace NAMESPACE {
void MAIN {
    // Compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);  // Tiles per row (width / 32)
    constexpr uint32_t Ht = get_compile_time_arg_val(1);  // Tile rows (height / 32)
    constexpr uint32_t W = get_compile_time_arg_val(2);   // Width in elements

    // Runtime args
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    // CB indices
    constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;        // Input RM sticks
    constexpr uint32_t cb_in_tiled = tt::CBIndex::c_1;     // Tiled input (double-buffered for reuse)
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;       // Scaler (1/W)
    constexpr uint32_t cb_eps = tt::CBIndex::c_3;          // Epsilon
    constexpr uint32_t cb_gamma_tiled = tt::CBIndex::c_6;  // Tiled gamma (PERSISTENT, reader fills directly)
    constexpr uint32_t cb_beta_tiled = tt::CBIndex::c_7;   // Tiled beta (PERSISTENT, reader fills directly)
    constexpr uint32_t cb_out_rm = tt::CBIndex::c_16;      // Output RM sticks
    constexpr uint32_t cb_centered = tt::CBIndex::c_24;    // x - mean (also used for squared, normalized)
    constexpr uint32_t cb_mean = tt::CBIndex::c_25;        // Mean tile
    constexpr uint32_t cb_var = tt::CBIndex::c_26;         // Variance tile / inv_std
    constexpr uint32_t cb_invstd = tt::CBIndex::c_27;      // Temporary for rsqrt computation

    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t onetile = 1;

    // Initialize compute kernel hardware - MUST be called before any compute operations
    compute_kernel_hw_startup(cb_in_rm, cb_in_tiled);

    // Wait for persistent data: scaler, epsilon, gamma, beta (pushed by reader once)
    cb_wait_front(cb_scaler, onetile);
    cb_wait_front(cb_eps, onetile);
    cb_wait_front(cb_gamma_tiled, Wt);
    cb_wait_front(cb_beta_tiled, Wt);

    // Per-row loop
    for (uint32_t row = 0; row < num_tile_rows; row++) {
        // =====================================================================
        // Phase 1: Tilize input (c_0 -> c_1)
        // Reader pushes 32 sticks to c_0, we tilize to c_1
        // =====================================================================
        cb_wait_front(cb_in_rm, TILE_HEIGHT);
        cb_reserve_back(cb_in_tiled, Wt);
        tilize_init(cb_in_rm, Wt, cb_in_tiled);
        tilize_block(cb_in_rm, Wt, cb_in_tiled);
        tilize_uninit(cb_in_rm, cb_in_tiled);
        cb_push_back(cb_in_tiled, Wt);
        cb_pop_front(cb_in_rm, TILE_HEIGHT);

        // At this point cb_in_tiled has Wt tiles of tiled input

        // =====================================================================
        // Phase 2: Compute Mean using PERSISTENT reduce
        // PERSISTENT mode: waits for tiles, does indexed access, does NOT pop
        // This keeps tiles in cb_in_tiled for subsequent centering operation
        // =====================================================================
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputMode::PERSISTENT>(
                cb_in_tiled, cb_scaler, cb_mean, compute_kernel_lib::TileShape::row(Wt));

        // cb_in_tiled tiles are still available (PERSISTENT mode didn't pop them)
        // cb_mean has 1 tile with the mean value (replicated across all elements)

        // =====================================================================
        // Phase 3: Subtract mean (center values): centered = x - mean
        // Use PERSISTENT mode for input A (cb_in_tiled) since we need tiles again
        // STREAMING for B (mean) since it's consumed
        // =====================================================================
        // We need to manually handle this since binary helper will pop both inputs
        // Let's use raw calls for centering to keep cb_in_tiled alive
        {
            sub_bcast_cols_init_short(cb_in_tiled, cb_mean);
            for (uint32_t t = 0; t < Wt; t++) {
                tile_regs_acquire();
                // cb_in_tiled tiles already waited from PERSISTENT reduce
                // cb_mean has 1 tile
                sub_tiles_bcast_cols(cb_in_tiled, cb_mean, t, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_centered, onetile);
                pack_tile(0, cb_centered);
                cb_push_back(cb_centered, onetile);
                tile_regs_release();
            }
        }
        // Now cb_centered has Wt tiles of (x - mean)
        // cb_in_tiled still has the original tiles (not popped)

        // =====================================================================
        // Phase 4: Square centered values for variance
        // Read from cb_centered, output to cb_in_tiled (reuse after we pop original)
        // =====================================================================
        // First pop the original input tiles from cb_in_tiled since we're done with them
        cb_pop_front(cb_in_tiled, Wt);
        // Pop mean tile since we're done with it
        cb_pop_front(cb_mean, onetile);

        {
            mul_tiles_init(cb_centered, cb_centered);
            for (uint32_t t = 0; t < Wt; t++) {
                tile_regs_acquire();
                cb_wait_front(cb_centered, onetile);
                mul_tiles(cb_centered, cb_centered, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_in_tiled, onetile);
                pack_tile(0, cb_in_tiled);
                cb_push_back(cb_in_tiled, onetile);
                tile_regs_release();
                cb_pop_front(cb_centered, onetile);
            }
        }
        // Now cb_in_tiled has squared values, cb_centered is empty

        // =====================================================================
        // Phase 5: Compute Variance = reduce_row(squares) * scaler
        // =====================================================================
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_in_tiled, cb_scaler, cb_var, compute_kernel_lib::TileShape::row(Wt));

        // cb_in_tiled (squares) consumed, cb_var has variance (1 tile)

        // =====================================================================
        // Phase 6: Add epsilon and compute rsqrt: inv_std = 1/sqrt(var + eps)
        // =====================================================================
        {
            // Add epsilon using scalar broadcast (epsilon fills entire tile)
            add_bcast_scalar_init_short(cb_var, cb_eps);
            tile_regs_acquire();
            cb_wait_front(cb_var, onetile);
            add_tiles_bcast_scalar(cb_var, cb_eps, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_invstd, onetile);
            pack_tile(0, cb_invstd);
            cb_push_back(cb_invstd, onetile);
            tile_regs_release();
            cb_pop_front(cb_var, onetile);

            // Compute rsqrt
            rsqrt_tile_init();
            tile_regs_acquire();
            cb_wait_front(cb_invstd, onetile);
            copy_tile_to_dst_init_short(cb_invstd);
            copy_tile(cb_invstd, 0, 0);
            rsqrt_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_var, onetile);
            pack_tile(0, cb_var);
            cb_push_back(cb_var, onetile);
            tile_regs_release();
            cb_pop_front(cb_invstd, onetile);
        }
        // Now cb_var has 1/sqrt(var+eps)

        // =====================================================================
        // Phase 7: Re-compute centered values since cb_centered was consumed
        // Need to re-tilize input from cb_in_rm... but we already popped it!
        // This is a design issue - we need the reader to push input multiple times
        // OR we need more CB capacity to keep both centered values
        //
        // For now, let's adjust the flow to keep centered values around
        // =====================================================================
        // DESIGN REVISION: We need to keep centered values for normalization
        // Let's restart with a better CB utilization strategy

        // Actually, wait - we consumed cb_centered during squaring.
        // We need (x - mean) for the normalize step too.
        // Options:
        // A) Have reader push input 3 times per row
        // B) Keep centered values in a separate persistent CB
        // C) Compute centered values twice (wasteful but simple)

        // Going with C for simplicity - re-tilize and re-center
        // This requires the reader to push the same input row multiple times
        // OR we modify the factory to have larger CBs

        // For the CURRENT factory configuration, cb_in_rm is popped after tilize.
        // We need to restructure the reader/compute contract.

        // TEMPORARY FIX: Assume reader pushes input 3 times per row
        // (We'll update the reader next)

        // Re-tilize input (reader should have pushed another copy)
        cb_wait_front(cb_in_rm, TILE_HEIGHT);
        cb_reserve_back(cb_in_tiled, Wt);
        tilize_init(cb_in_rm, Wt, cb_in_tiled);
        tilize_block(cb_in_rm, Wt, cb_in_tiled);
        tilize_uninit(cb_in_rm, cb_in_tiled);
        cb_push_back(cb_in_tiled, Wt);
        cb_pop_front(cb_in_rm, TILE_HEIGHT);

        // Re-compute mean
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputMode::PERSISTENT>(
                cb_in_tiled, cb_scaler, cb_mean, compute_kernel_lib::TileShape::row(Wt));

        // Re-compute centered values
        {
            sub_bcast_cols_init_short(cb_in_tiled, cb_mean);
            for (uint32_t t = 0; t < Wt; t++) {
                tile_regs_acquire();
                sub_tiles_bcast_cols(cb_in_tiled, cb_mean, t, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_centered, onetile);
                pack_tile(0, cb_centered);
                cb_push_back(cb_centered, onetile);
                tile_regs_release();
            }
        }
        cb_pop_front(cb_in_tiled, Wt);
        cb_pop_front(cb_mean, onetile);

        // =====================================================================
        // Phase 8: Normalize: normalized = centered * inv_std
        // Broadcast inv_std (cb_var, 1 tile) across all columns
        // =====================================================================
        compute_kernel_lib::mul<compute_kernel_lib::BroadcastDim::COL>(
            cb_centered, cb_var, cb_in_tiled, compute_kernel_lib::BinaryTileShape::row(Wt));

        // cb_centered and cb_var consumed, cb_in_tiled has normalized values

        // =====================================================================
        // Phase 9: Apply gamma: scaled = normalized * gamma
        // ROW broadcast - gamma (Wt tiles) applied to each row
        // gamma is PERSISTENT (not popped by helper)
        // =====================================================================
        // The ROW broadcast helper waits for Wt tiles of B but doesn't pop
        // Actually, looking at the helper code, ROW broadcast does wait/not-pop B
        compute_kernel_lib::mul<compute_kernel_lib::BroadcastDim::ROW>(
            cb_in_tiled, cb_gamma_tiled, cb_centered, compute_kernel_lib::BinaryTileShape::row(Wt));

        // cb_in_tiled consumed, cb_centered has scaled values
        // cb_gamma_tiled NOT consumed (persists)

        // =====================================================================
        // Phase 10: Apply beta: output = scaled + beta
        // ROW broadcast - beta (Wt tiles) applied to each row
        // beta is PERSISTENT (not popped by helper)
        // =====================================================================
        compute_kernel_lib::add<compute_kernel_lib::BroadcastDim::ROW>(
            cb_centered, cb_beta_tiled, cb_in_tiled, compute_kernel_lib::BinaryTileShape::row(Wt));

        // cb_centered consumed, cb_in_tiled has final output
        // cb_beta_tiled NOT consumed (persists)

        // =====================================================================
        // Phase 11: Untilize output
        // =====================================================================
        cb_wait_front(cb_in_tiled, Wt);
        cb_reserve_back(cb_out_rm, TILE_HEIGHT);
        pack_untilize_init<Wt, Wt>(cb_in_tiled, cb_out_rm);
        pack_untilize_block<Wt, Wt>(cb_in_tiled, 1, cb_out_rm);
        pack_untilize_uninit(cb_out_rm);
        cb_push_back(cb_out_rm, TILE_HEIGHT);
        cb_pop_front(cb_in_tiled, Wt);
    }

    // Persistent CBs (scaler, eps, gamma, beta) are never popped
}
}  // namespace NAMESPACE
