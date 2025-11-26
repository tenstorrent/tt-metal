// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/cb_api.h"

/**
 * @file untilize_helpers.h
 * @brief Single unified untilize function with automatic dispatch
 *
 * Provides ONE function that handles all untilize operations:
 * - Small widths (≤ DEST limit): Uses pack_untilize (hardware-accelerated, preferred)
 * - Large widths (> DEST limit): Uses standard untilize (fallback for widths exceeding hardware limit)
 *
 * DEST register capacity is automatically detected from JIT-generated headers:
 * - DST_SYNC_MODE (Half/Full sync mode)
 * - DST_ACCUM_MODE (16-bit/32-bit accumulation)
 *
 * IMPORTANT: Requires compute kernel hardware initialization.
 * Call compute_kernel_hw_startup() before using.
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.h"
 *
 *   compute_kernel_hw_startup(cb_in, cb_out);
 *
 *   // Small width - automatically uses pack_untilize (hardware-accelerated)
 *   compute_kernel_lib::untilize<4>(cb_in, cb_out, num_rows);
 *
 *   // Large width - automatically uses standard untilize (fallback)
 *   compute_kernel_lib::untilize<32>(cb_in, cb_out, num_rows);
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
// Unified Init/Uninit Functions
// =============================================================================

/**
 * @brief Initialize untilize - automatically dispatches based on width
 *
 * @tparam tile_width Width in tiles
 * @param icb Input circular buffer ID
 * @param ocb Output circular buffer ID (only used for pack path)
 */
template <uint32_t tile_width>
ALWI void untilize_init(uint32_t icb, uint32_t ocb = 0) {
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    if constexpr (tile_width > dest_limit) {
        ::untilize_init(icb);
    } else {
        pack_untilize_init<tile_width, tile_width>(icb, ocb);
    }
}

/**
 * @brief Uninitialize untilize - automatically dispatches based on width
 *
 * @tparam tile_width Width in tiles
 * @param icb Input circular buffer ID (only used for standard path)
 * @param ocb Output circular buffer ID (only used for pack path)
 */
template <uint32_t tile_width>
ALWI void untilize_uninit(uint32_t icb = 0, uint32_t ocb = 0) {
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    if constexpr (tile_width > dest_limit) {
        ::untilize_uninit(icb);
    } else {
        pack_untilize_uninit(ocb);
    }
}

// =============================================================================
// Single Unified Untilize Function
// =============================================================================

/**
 * @brief Unified untilize function - automatically dispatches based on width
 *
 * This is the ONLY untilize function you need. Provide the tile width,
 * and the optimal implementation is selected at compile time based on
 * auto-detected DEST register capacity.
 *
 * Dispatch logic:
 * - tile_width <= DEST capacity AND !wait_upfront: Pack untilize (hardware-accelerated, preferred)
 * - tile_width > DEST capacity OR wait_upfront: Standard untilize (fallback)
 *
 * pack_untilize has a hardware limit (~8 tiles) and cannot handle larger widths
 * or wait_upfront pattern. For these cases, we fall back to standard untilize which
 * can handle any width by processing tiles sequentially.
 *
 * @tparam tile_width Width in tiles (number of tiles per row)
 * @tparam init Call init before processing (default: true)
 * @tparam uninit Call uninit after processing (default: true)
 * @tparam wait_upfront Wait for all tiles upfront instead of per-row (default: false)
 *                      Used by GroupNorm and similar operations.
 *                      Forces use of standard untilize path (pack_untilize doesn't support this).
 *
 * @param icb Input circular buffer ID (tiled data)
 * @param ocb Output circular buffer ID (row-major data)
 * @param num_rows Number of rows to process
 * @param block_rt_dim Row height per block in tiles (default: 1)
 *                     Only used in pack_untilize path for multi-row blocks.
 *                     Standard path always processes single-row blocks.
 * @param total_tiles Total tiles to wait for when wait_upfront=true (default: 0 = auto-compute)
 *
 * @example
 *   // Width 4 - uses pack_untilize (hardware-accelerated)
 *   untilize<4>(cb_in, cb_out, 10);
 *
 * @example
 *   // Width 32 - automatically uses standard untilize (fallback)
 *   untilize<32>(cb_in, cb_out, 10);
 *
 * @example
 *   // Wait-upfront pattern (forces standard untilize regardless of width)
 *   untilize<10, true, true, true>(cb_in, cb_out, num_rows, 1, total_tiles);
 *
 * @example
 *   // Skip init for reinit scenarios
 *   untilize<8, false>(cb_in, cb_out, 10);
 */
template <uint32_t tile_width, bool init = true, bool uninit = true, bool wait_upfront = false>
ALWI void untilize(uint32_t icb, uint32_t ocb, uint32_t num_rows, uint32_t block_rt_dim = 1, uint32_t total_tiles = 0) {
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;

    // wait_upfront pattern uses standard path because pack_untilize doesn't support it
    if constexpr (wait_upfront || tile_width > dest_limit) {
        // =================================================================
        // STANDARD UNTILIZE PATH
        // Width exceeds DEST limit OR wait_upfront - use standard untilize (fallback)
        // Standard untilize can handle any width by processing tiles sequentially
        // =================================================================

        if constexpr (init) {
            ::untilize_init(icb);
        }

        if constexpr (wait_upfront) {
            // Wait for all tiles upfront
            uint32_t wait_amount = (total_tiles > 0) ? total_tiles : (tile_width * num_rows);
            cb_wait_front(icb, wait_amount);
        }

        for (uint32_t r = 0; r < num_rows; ++r) {
            if constexpr (!wait_upfront) {
                cb_wait_front(icb, tile_width);
            }
            cb_reserve_back(ocb, tile_width);
            untilize_block(icb, tile_width, ocb);
            cb_push_back(ocb, tile_width);
            cb_pop_front(icb, tile_width);
        }

        if constexpr (uninit) {
            ::untilize_uninit(icb);
        }

    } else {
        // =================================================================
        // PACK UNTILIZE PATH
        // Width fits in DEST - use hardware-accelerated pack_untilize (preferred)
        // pack_untilize has hardware limit of ~8 tiles, cannot handle larger widths
        // Note: wait_upfront is handled by standard path above
        // =================================================================

        if constexpr (init) {
            pack_untilize_init<tile_width, tile_width>(icb, ocb);
        }

        const uint32_t tiles_per_row = tile_width * block_rt_dim;

        for (uint32_t r = 0; r < num_rows; ++r) {
            cb_wait_front(icb, tiles_per_row);
            cb_reserve_back(ocb, tiles_per_row);
            pack_untilize_block<tile_width, tile_width>(icb, block_rt_dim, ocb, 0);
            cb_pop_front(icb, tiles_per_row);
            cb_push_back(ocb, tiles_per_row);
        }

        if constexpr (uninit) {
            pack_untilize_uninit(ocb);
        }
    }
}

}  // namespace compute_kernel_lib
