// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

/**
 * @file untilize_helpers.hpp
 * @brief Single unified untilize function with automatic dispatch
 *
 * Provides ONE function that handles all untilize operations:
 * - Small widths (≤ DEST limit): Uses pack_untilize (hardware-accelerated, preferred)
 * - Large widths (> DEST limit) with integer types: Uses block-based pack_untilize (hardware-accelerated)
 * - Large widths (> DEST limit) with non-integer types: Uses standard untilize (fallback)
 *
 * DEST register capacity is automatically detected via dest_helpers.hpp.
 *
 * Data format is automatically detected from JIT-generated header:
 * - unpack_dst_format[cb_id] contains the DataFormat enum value
 *
 * IMPORTANT: Requires compute kernel hardware initialization.
 * Call compute_kernel_hw_startup() before using.
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
 *
 *   compute_kernel_hw_startup(cb_in, cb_out);
 *
 *   // Small width - automatically uses pack_untilize (hardware-accelerated)
 *   compute_kernel_lib::untilize<4>(cb_in, cb_out, num_rows);
 *
 *   // Large width with integer - automatically uses block-based pack_untilize (hardware-accelerated)
 *   compute_kernel_lib::untilize<32>(cb_in, cb_out, num_rows);
 *
 *   // Large width with float - automatically uses standard untilize (fallback)
 *   compute_kernel_lib::untilize<32>(cb_in, cb_out, num_rows);
 */

namespace compute_kernel_lib {

// get_dest_limit() and DEST_AUTO_LIMIT are provided by dest_helpers.hpp

// =============================================================================
// Data Format Detection - Automatic Detection
// =============================================================================

// unpack_dst_format is defined in JIT-generated chlkc_unpack_data_format.h
// It's an array where unpack_dst_format[cb_id] contains the DataFormat enum value

// Integer data formats from tt_metal/hw/inc/tt-1xx/blackhole/tensix_types.h:
// - Int8 = 14
// - UInt8 = 30
// - UInt16 = 9
// - Int32 = 8
// - UInt32 = 24

template <uint32_t cb_id>
constexpr bool is_integer_format() {
// Check if unpack_dst_format array is available (from JIT-generated chlkc_unpack_data_format.h)
// This header is included via chlkc_list.h in the firmware build
#if __has_include("chlkc_unpack_data_format.h")
// Include the JIT-generated header
#include "chlkc_unpack_data_format.h"

    // Access the format at compile time
    constexpr uint32_t format = unpack_dst_format[cb_id];

    // Check if format is one of the integer types
    // Integer formats from tt_metal/hw/inc/tt-1xx/blackhole/tensix_types.h:
    return format == 8 ||   // Int32
           format == 24 ||  // UInt32
           format == 14 ||  // Int8
           format == 30 ||  // UInt8
           format == 9;     // UInt16
#else
    // If header not available, assume integer for wide widths (conservative for pack_untilize)
    // This ensures wide integer tensors get hardware acceleration
    return true;  // Changed to true - prefer pack_untilize block-based path
#endif
}

// =============================================================================
// Block Splitting Helper for Wide Integer Untilize
// =============================================================================

/**
 * @brief Compute number of blocks needed to split a wide row into DEST-sized chunks
 *
 * Finds the largest divisor of total_width that is <= max_block_width
 * This ensures optimal block size while respecting DEST register limits.
 *
 * @param total_width Total width in tiles to be split
 * @param max_block_width Maximum block width (DEST register limit)
 * @return Number of blocks needed
 */
constexpr uint32_t compute_num_blocks(uint32_t total_width, uint32_t max_block_width) {
    for (uint32_t block_width = max_block_width; block_width >= 1; --block_width) {
        if (total_width % block_width == 0) {
            return total_width / block_width;
        }
    }
    return total_width;  // fallback: 1 tile per block
}

// =============================================================================
// Unified Init/Uninit Functions
// =============================================================================

/**
 * @brief Initialize untilize - automatically dispatches based on width and data format
 *
 * @tparam tile_width Width in tiles
 * @param icb Input circular buffer ID
 * @param ocb Output circular buffer ID (only used for pack path)
 */
template <uint32_t tile_width, uint32_t icb, uint32_t ocb = 0>
ALWI void untilize_init() {
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool is_integer = is_integer_format<icb>();

    if constexpr (tile_width > dest_limit && is_integer) {
        // Integer with wide width - use block-based pack_untilize
        constexpr uint32_t num_blocks = compute_num_blocks(tile_width, dest_limit);
        constexpr uint32_t block_width = tile_width / num_blocks;
        pack_untilize_init<block_width, tile_width>(icb, ocb);
    } else if constexpr (tile_width > dest_limit) {
        // Non-integer with wide width - use standard untilize
        ::untilize_init(icb);
    } else {
        // Narrow width - use pack_untilize
        pack_untilize_init<tile_width, tile_width>(icb, ocb);
    }
}

/**
 * @brief Uninitialize untilize - automatically dispatches based on width and data format
 *
 * @tparam tile_width Width in tiles
 * @param icb Input circular buffer ID (only used for standard path)
 * @param ocb Output circular buffer ID (only used for pack path)
 */
template <uint32_t tile_width, uint32_t icb = 0, uint32_t ocb = 0>
ALWI void untilize_uninit() {
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool is_integer = is_integer_format<icb>();

    if constexpr (tile_width > dest_limit && !is_integer) {
        // Non-integer with wide width - standard untilize path
        ::untilize_uninit(icb);
    } else {
        // Pack untilize path (for narrow widths or wide integers)
        pack_untilize_uninit(ocb);
    }
}

// =============================================================================
// Single Unified Untilize Function
// =============================================================================

/**
 * @brief Unified untilize function - automatically dispatches based on width, data format, and pattern
 *
 * This is the ONLY untilize function you need. Provide the tile width and CB IDs,
 * and the optimal implementation is selected at compile time based on:
 * 1. Auto-detected DEST register capacity
 * 2. Auto-detected data format (integer vs non-integer)
 * 3. Width constraints
 *
 * Dispatch logic:
 * - tile_width <= DEST capacity: Pack untilize (hardware-accelerated, single-pass)
 * - tile_width > DEST capacity AND integer type: Block-based pack untilize (hardware-accelerated, multi-pass)
 * - tile_width > DEST capacity AND non-integer: Standard untilize (fallback for float types)
 * - wait_upfront=true: Always use standard untilize (pack_untilize doesn't support this pattern)
 *
 * Integer data types (Int8, UInt8, UInt16, Int32, UInt32) can use block-based pack_untilize
 * even for wide widths, providing hardware acceleration by splitting the row into blocks
 * that each fit within DEST register limits.
 *
 * @tparam tile_width Width in tiles (number of tiles per row)
 * @tparam icb_id Input circular buffer ID (tiled data) - must be compile-time constant
 * @tparam ocb_id Output circular buffer ID (row-major data) - must be compile-time constant
 * @tparam init Call init before processing (default: true)
 * @tparam uninit Call uninit after processing (default: true)
 * @tparam wait_upfront Wait for all tiles upfront instead of per-row (default: false)
 *                      Used by GroupNorm and similar operations.
 *                      Forces use of standard untilize path.
 *
 * @param num_rows Number of rows to process
 * @param block_rt_dim Row height per block in tiles (default: 1)
 *                     Only used in pack_untilize path for multi-row blocks.
 *                     Standard path always processes single-row blocks.
 * @param total_tiles Total tiles to wait for when wait_upfront=true (default: 0 = auto-compute)
 *
 * @example
 *   // Width 4 - uses pack_untilize (hardware-accelerated, single-pass)
 *   untilize<4, cb_in, cb_out>(10);
 *
 * @example
 *   // Width 32 with INT32 - automatically uses block-based pack_untilize (hardware-accelerated)
 *   untilize<32, cb_in, cb_out>(1);
 *
 * @example
 *   // Width 32 with Float16 - automatically uses standard untilize (fallback)
 *   untilize<32, cb_in, cb_out>(10);
 *
 * @example
 *   // Wait-upfront pattern (forces standard untilize regardless of width/type)
 *   untilize<10, cb_in, cb_out, true, true, true>(num_rows, 1, total_tiles);
 */
template <
    uint32_t tile_width,
    uint32_t icb_id,
    uint32_t ocb_id,
    bool init = true,
    bool uninit = true,
    bool wait_upfront = false>
ALWI void untilize(uint32_t num_rows, uint32_t block_rt_dim = 1, uint32_t total_tiles = 0) {
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool is_integer = is_integer_format<icb_id>();

    // wait_upfront pattern always uses standard path because pack_untilize doesn't support it
    if constexpr (wait_upfront || (tile_width > dest_limit && !is_integer)) {
        // =================================================================
        // STANDARD UNTILIZE PATH
        // Used when:
        // - wait_upfront=true (GroupNorm pattern)
        // - Width exceeds DEST AND non-integer type (float fallback)
        // =================================================================

        if constexpr (init) {
            ::untilize_init(icb_id);
        }

        if constexpr (wait_upfront) {
            // Wait for all tiles upfront
            uint32_t wait_amount = (total_tiles > 0) ? total_tiles : (tile_width * num_rows);
            cb_wait_front(icb_id, wait_amount);
        }

        for (uint32_t r = 0; r < num_rows; ++r) {
            if constexpr (!wait_upfront) {
                cb_wait_front(icb_id, tile_width);
            }
            cb_reserve_back(ocb_id, tile_width);
            untilize_block(icb_id, tile_width, ocb_id);
            cb_push_back(ocb_id, tile_width);
            cb_pop_front(icb_id, tile_width);
        }

        if constexpr (uninit) {
            ::untilize_uninit(icb_id);
        }

    } else if constexpr (tile_width > dest_limit && is_integer) {
        // =================================================================
        // BLOCK-BASED PACK UNTILIZE PATH
        // Used for integer types with width exceeding DEST limit
        // Splits wide rows into multiple blocks that each fit in DEST
        // Provides hardware acceleration for wide integer tensors
        // =================================================================

        constexpr uint32_t num_blocks = compute_num_blocks(tile_width, dest_limit);
        constexpr uint32_t block_width = tile_width / num_blocks;

        if constexpr (init) {
            pack_untilize_init<block_width, tile_width>(icb_id, ocb_id);
        }

        for (uint32_t r = 0; r < num_rows; ++r) {
            cb_reserve_back(ocb_id, tile_width);
            for (uint32_t b = 0; b < num_blocks; ++b) {
                cb_wait_front(icb_id, block_width);
                pack_untilize_block<block_width, tile_width>(icb_id, 1, ocb_id, b);
                cb_pop_front(icb_id, block_width);
            }
            cb_push_back(ocb_id, tile_width);
        }

        if constexpr (uninit) {
            pack_untilize_uninit(ocb_id);
        }

    } else {
        // =================================================================
        // PACK UNTILIZE PATH (SINGLE-PASS)
        // Used when width fits in DEST (optimal for all data types)
        // =================================================================

        if constexpr (init) {
            pack_untilize_init<tile_width, tile_width>(icb_id, ocb_id);
        }

        const uint32_t tiles_per_row = tile_width * block_rt_dim;

        for (uint32_t r = 0; r < num_rows; ++r) {
            cb_wait_front(icb_id, tiles_per_row);
            cb_reserve_back(ocb_id, tiles_per_row);
            pack_untilize_block<tile_width, tile_width>(icb_id, block_rt_dim, ocb_id, 0);
            cb_pop_front(icb_id, tiles_per_row);
            cb_push_back(ocb_id, tiles_per_row);
        }

        if constexpr (uninit) {
            pack_untilize_uninit(ocb_id);
        }
    }
}

}  // namespace compute_kernel_lib
