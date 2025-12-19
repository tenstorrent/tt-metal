// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"

/**
 * @file reduce_helpers.h
 * @brief Single unified reduce function with automatic dispatch
 *
 * Provides ONE function that handles all reduce operations:
 * - Row reduction (REDUCE_ROW): Reduces W dimension, outputs Ht tiles per batch
 * - Column reduction (REDUCE_COL): Reduces H dimension, outputs Wt tiles per batch
 * - Scalar reduction (REDUCE_SCALAR): Reduces both H and W, outputs 1 tile per batch
 *
 * This library hides the complexity of:
 * - tile_regs_acquire/commit/wait/release DST register management
 * - reduce_init/reduce_uninit initialization
 * - Circular buffer manipulation (cb_wait_front, cb_pop_front, cb_reserve_back, cb_push_back)
 * - pack_tile for writing results to output CB
 *
 * DEST register capacity is automatically detected from JIT-generated headers:
 * - DST_SYNC_MODE (Half/Full sync mode)
 * - DST_ACCUM_MODE (16-bit/32-bit accumulation)
 *
 * IMPORTANT: Requires compute kernel hardware initialization.
 * Call compute_kernel_hw_startup() before using.
 *
 * IMPORTANT: The scaler CB must contain the scaling factor tile BEFORE calling reduce().
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.h"
 *
 *   compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
 *
 *   // Reduce entire HxW grid to single tile (REDUCE_SCALAR)
 *   compute_kernel_lib::reduce<SUM, REDUCE_SCALAR>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 *   // Reduce each row (W dimension) - output has Ht tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 *   // Reduce each column (H dimension) - output has Wt tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 *   // Using defines for reduce type/dim (REDUCE_OP and REDUCE_DIM must be defined)
 *   compute_kernel_lib::reduce(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 */

namespace compute_kernel_lib {

// =============================================================================
// DEST Register Capacity - Automatic Detection
// =============================================================================

// DST_SYNC_MODE is defined in JIT-generated chlkc_dst_sync_mode.h
// DST_ACCUM_MODE is defined in JIT-generated chlkc_dst_accum_mode.h
// Both are included via chlkc_list.h -> common_globals.h

// DEST register capacity depends on:
// 1. Sync mode (Half vs Full) - determined by DST_SYNC_MODE
// 2. Accumulation mode (16-bit vs 32-bit) - determined by DST_ACCUM_MODE
//
// Capacity table:
// - SyncFull + 16-bit (DST_ACCUM_MODE=false): 16 tiles
// - SyncFull + 32-bit (DST_ACCUM_MODE=true):  8 tiles
// - SyncHalf + 16-bit (DST_ACCUM_MODE=false): 8 tiles
// - SyncHalf + 32-bit (DST_ACCUM_MODE=true):  4 tiles

constexpr uint32_t get_dest_limit() {
#if defined(DST_SYNC_MODE) && defined(DST_ACCUM_MODE)
    // Automatically detect from JIT-generated header files
    if constexpr (DST_SYNC_MODE == DstSync::SyncFull) {
        // Full-sync mode
        if constexpr (DST_ACCUM_MODE) {
            return 8;  // 32-bit accumulation
        } else {
            return 16;  // 16-bit accumulation
        }
    } else {
        // Half-sync mode
        if constexpr (DST_ACCUM_MODE) {
            return 4;  // 32-bit accumulation
        } else {
            return 8;  // 16-bit accumulation
        }
    }
#else
    // Fallback if JIT headers not defined (shouldn't happen in real kernels)
    // Use conservative half-sync 16-bit value
    return 8;
#endif
}

// Auto-detected default dest limit based on current sync and accumulation modes
constexpr uint32_t DEST_AUTO_LIMIT = get_dest_limit();

// =============================================================================
// Single Unified Reduce Function
// =============================================================================

/**
 * @brief Unified reduce function handling all reduction patterns
 *
 * This single function handles:
 * - Row reduction (REDUCE_ROW): Reduces W dimension, outputs Ht tiles per batch
 * - Column reduction (REDUCE_COL): Reduces H dimension, outputs Wt tiles per batch
 * - Scalar reduction (REDUCE_SCALAR): Reduces both H and W, outputs 1 tile per batch
 *
 * IMPORTANT - HARDWARE INITIALIZATION REQUIREMENT:
 * Before calling this function, you MUST initialize the compute kernel hardware by
 * calling compute_kernel_hw_startup() at the start of your kernel.
 *
 * IMPORTANT - SCALER CB REQUIREMENT:
 * The scaler CB (icb_scaler) must contain the scaling factor tile BEFORE calling
 * this function. The function will wait for it automatically when init=true.
 *
 * IMPORTANT - REDUCE_COL DATA LAYOUT:
 * For REDUCE_COL, tiles must arrive in N C W_skip H W_chunk order (chunked by row_chunk).
 * If the host provides a specific row_chunk, pass it to ensure correct data interpretation.
 * If row_chunk=0 (default), the auto-detected DEST limit is used.
 *
 * @tparam reduce_type The type of reduce operation (SUM, AVG, MAX) - defaults to REDUCE_OP define
 * @tparam reduce_dim The dimension to reduce (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR) - defaults to REDUCE_DIM define
 * @tparam init If true, calls reduce_init before processing (default: true)
 * @tparam uninit If true, calls reduce_uninit after processing (default: true)
 * @tparam enforce_fp32_accumulation Enable FP32 accumulation (default: false)
 *
 * @param icb Input circular buffer containing tiles to reduce
 * @param icb_scaler Circular buffer containing scaler tile
 * @param ocb Output circular buffer for reduced tiles
 * @param Ht Height in tiles (number of tile rows)
 * @param Wt Width in tiles (number of tile columns)
 * @param num_batches Number of batches to process (NC dimension)
 * @param row_chunk Chunk size for REDUCE_COL (default: 0 = use auto-detected DEST limit)
 *                  For REDUCE_ROW and REDUCE_SCALAR, this parameter is ignored.
 *                  For REDUCE_COL, if the host arranges tiles with a specific chunk size,
 *                  pass that value here to ensure correct data interpretation.
 *
 * @example
 *   // Reduce entire HxW grid to single tile (REDUCE_SCALAR)
 *   compute_kernel_lib::reduce<SUM, REDUCE_SCALAR>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 * @example
 *   // Reduce each row (W dimension) - output has Ht tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 * @example
 *   // Reduce each column (H dimension) - output has Wt tiles per batch, auto chunk size
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL>(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 *
 * @example
 *   // Reduce each column with host-specified chunk size
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL>(cb_in, cb_scaler, cb_out, Ht, Wt, NC, row_chunk);
 *
 * @example
 *   // Using defines for reduce type/dim
 *   compute_kernel_lib::reduce(cb_in, cb_scaler, cb_out, Ht, Wt, NC);
 */
template <
    PoolType reduce_type = REDUCE_OP,
    ReduceDim reduce_dim = REDUCE_DIM,
    bool init = true,
    bool uninit = true,
    bool enforce_fp32_accumulation = false>
ALWI void reduce(
    uint32_t icb,
    uint32_t icb_scaler,
    uint32_t ocb,
    uint32_t Ht,
    uint32_t Wt,
    uint32_t num_batches,
    uint32_t row_chunk = 0) {
    // Initialization
    if constexpr (init) {
        reduce_init<reduce_type, reduce_dim, enforce_fp32_accumulation>(icb, icb_scaler, ocb);
        cb_wait_front(icb_scaler, 1);  // Wait for scaler tile
    }

    constexpr uint32_t onetile = 1;

    // Pattern dispatch based on reduce_dim
    if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
        // =================================================================
        // REDUCE_SCALAR: HW reduction - all tiles -> 1 output tile per batch
        // =================================================================
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            tile_regs_acquire();
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    cb_wait_front(icb, onetile);
                    reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(icb, icb_scaler, 0, 0, 0);
                    cb_pop_front(icb, onetile);
                }
            }
            cb_reserve_back(ocb, onetile);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, ocb);
            tile_regs_release();
            cb_push_back(ocb, onetile);
        }
    } else if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
        // =================================================================
        // REDUCE_ROW: W reduction - each row -> 1 output tile (Ht outputs per batch)
        // =================================================================
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                tile_regs_acquire();
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    cb_wait_front(icb, onetile);
                    reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(icb, icb_scaler, 0, 0, 0);
                    cb_pop_front(icb, onetile);
                }
                cb_reserve_back(ocb, onetile);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, ocb);
                tile_regs_release();
                cb_push_back(ocb, onetile);
            }
        }
    } else {
        // =================================================================
        // REDUCE_COL: H reduction - each column -> 1 output tile (Wt outputs per batch)
        // Need chunking due to DEST register limits
        // Tiles arrive in N C W_skip H W_chunk order (chunked by chunk_size)
        // =================================================================

        // Use provided row_chunk if > 0, otherwise use auto-detected DEST limit
        const uint32_t chunk_size = (row_chunk > 0) ? row_chunk : DEST_AUTO_LIMIT;

        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t wt = 0; wt < Wt; wt += chunk_size) {
                uint32_t chunk_end = (wt + chunk_size < Wt) ? (wt + chunk_size) : Wt;
                uint32_t current_chunk = chunk_end - wt;

                tile_regs_acquire();
                for (uint32_t ht = 0; ht < Ht; ++ht) {
                    uint32_t dst_idx = 0;
                    for (uint32_t i = wt; i < chunk_end; ++i) {
                        cb_wait_front(icb, onetile);
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(icb, icb_scaler, 0, 0, dst_idx);
                        cb_pop_front(icb, onetile);
                        ++dst_idx;
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < current_chunk; ++i) {
                    cb_reserve_back(ocb, onetile);
                    pack_tile(i, ocb);
                    cb_push_back(ocb, onetile);
                }
                tile_regs_release();
            }
        }
    }

    // Cleanup
    if constexpr (uninit) {
        reduce_uninit<enforce_fp32_accumulation>();
    }
}

}  // namespace compute_kernel_lib
