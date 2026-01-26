// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/untilize.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/kernel_lib_types.hpp"

/**
 * @file untilize_helpers.hpp
 * @brief Single unified untilize function with automatic dispatch
 *
 * Provides ONE function that handles all untilize operations:
 * - Small widths (<= DEST limit): Uses pack_untilize (hardware-accelerated, preferred)
 * - Large widths (> DEST limit) with integer types: Uses block-based pack_untilize (hardware-accelerated)
 * - Large widths (> DEST limit) with non-integer types: Uses standard untilize (fallback)
 *
 * DEST register capacity is automatically detected via dest_helpers.hpp.
 *
 * IMPORTANT: Requires compute kernel hardware initialization.
 * Call compute_kernel_hw_startup() before using.
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
 *   using namespace compute_kernel_lib;
 *
 *   // Default behavior (most common)
 *   untilize<UntilizeConfig<WidthInTiles<4>, InputCB<cb_in>, OutputCB<cb_out>>>(num_rows);
 *
 *   // Wait upfront (GroupNorm pattern)
 *   untilize<UntilizeConfig<WidthInTiles<N>, InputCB<cb_in>, OutputCB<cb_out>,
 *                           UntilizeFlags::WAIT_UPFRONT>>(num_rows, 1, total_tiles);
 *
 *   // Skip init only
 *   untilize<UntilizeConfig<WidthInTiles<4>, InputCB<cb_in>, OutputCB<cb_out>,
 *                           UntilizeFlags::SKIP_INIT>>(num_rows);
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
    // If header not available, assume NOT integer (conservative for correctness)
    // This ensures non-integer formats like bfloat8_b use the standard untilize path
    // for wide tensors, which is required for correct results.
    return false;
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
 * @brief Initialize untilize - automatically dispatches based on Config
 * @tparam Config UntilizeConfig<WidthInTiles<N>, InputCB<N>, OutputCB<N>, Flags>
 */
template <typename Config>
ALWI void untilize_init() {
    static_assert(
        std::is_base_of_v<UntilizeConfigBase, Config>,
        "Config must derive from UntilizeConfigBase (use UntilizeConfig<WidthInTiles<N>, InputCB<N>, OutputCB<N>>)");

    constexpr uint32_t width_in_tiles = Config::width_in_tiles;
    constexpr uint32_t input_cb = Config::input_cb;
    constexpr uint32_t output_cb = Config::output_cb;
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool is_integer = is_integer_format<input_cb>();

    if constexpr (width_in_tiles > dest_limit && is_integer) {
        constexpr uint32_t num_blocks = compute_num_blocks(width_in_tiles, dest_limit);
        constexpr uint32_t block_w = width_in_tiles / num_blocks;
        pack_untilize_init<block_w, width_in_tiles>(input_cb, output_cb);
    } else if constexpr (width_in_tiles > dest_limit) {
        ::untilize_init(input_cb);
    } else {
        pack_untilize_init<width_in_tiles, width_in_tiles>(input_cb, output_cb);
    }
}

/**
 * @brief Uninitialize untilize - automatically dispatches based on Config
 * @tparam Config UntilizeConfig<WidthInTiles<N>, InputCB<N>, OutputCB<N>, Flags>
 */
template <typename Config>
ALWI void untilize_uninit() {
    static_assert(
        std::is_base_of_v<UntilizeConfigBase, Config>,
        "Config must derive from UntilizeConfigBase (use UntilizeConfig<WidthInTiles<N>, InputCB<N>, OutputCB<N>>)");

    constexpr uint32_t width_in_tiles = Config::width_in_tiles;
    constexpr uint32_t input_cb = Config::input_cb;
    constexpr uint32_t output_cb = Config::output_cb;
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool is_integer = is_integer_format<input_cb>();

    if constexpr (width_in_tiles > dest_limit && !is_integer) {
        ::untilize_uninit(input_cb);
    } else {
        pack_untilize_uninit(output_cb);
    }
}

// =============================================================================
// Single Unified Untilize Function
// =============================================================================

/**
 * @brief Unified untilize function with named parameters via Config struct
 *
 * @tparam Config UntilizeConfig<WidthInTiles<N>, InputCB<N>, OutputCB<N>, Flags>
 *
 * @param num_rows Number of rows to process
 * @param row_height_tiles Height per block in tiles (default: 1)
 * @param upfront_wait_tiles Total tiles for WAIT_UPFRONT (default: 0 = auto)
 *
 * @example
 *   // Default behavior
 *   untilize<UntilizeConfig<WidthInTiles<4>, InputCB<cb_in>, OutputCB<cb_out>>>(10);
 *
 * @example
 *   // With flags
 *   untilize<UntilizeConfig<WidthInTiles<N>, InputCB<cb_in>, OutputCB<cb_out>,
 *                           UntilizeFlags::WAIT_UPFRONT>>(num_rows, 1, total);
 */
template <typename Config>
ALWI void untilize(uint32_t num_rows, uint32_t row_height_tiles = 1, uint32_t upfront_wait_tiles = 0) {
    static_assert(
        std::is_base_of_v<UntilizeConfigBase, Config>,
        "Config must derive from UntilizeConfigBase (use UntilizeConfig<WidthInTiles<N>, InputCB<N>, OutputCB<N>>)");

    constexpr uint32_t width_in_tiles = Config::width_in_tiles;
    constexpr uint32_t input_cb = Config::input_cb;
    constexpr uint32_t output_cb = Config::output_cb;
    constexpr UntilizeFlags flags = Config::flags;

    constexpr bool do_init = !has_flag(flags, UntilizeFlags::SKIP_INIT);
    constexpr bool do_uninit = !has_flag(flags, UntilizeFlags::SKIP_UNINIT);
    constexpr bool wait_upfront = has_flag(flags, UntilizeFlags::WAIT_UPFRONT);

    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool is_integer = is_integer_format<input_cb>();

    if constexpr (wait_upfront || (width_in_tiles > dest_limit && !is_integer)) {
        // Standard untilize path (WAIT_UPFRONT or wide non-integer)
        if constexpr (do_init) {
            ::untilize_init(input_cb);
        }

        if constexpr (wait_upfront) {
            uint32_t wait_amount = (upfront_wait_tiles > 0) ? upfront_wait_tiles : (width_in_tiles * num_rows);
            cb_wait_front(input_cb, wait_amount);
        }

        for (uint32_t r = 0; r < num_rows; ++r) {
            if constexpr (!wait_upfront) {
                cb_wait_front(input_cb, width_in_tiles);
            }
            cb_reserve_back(output_cb, width_in_tiles);
            untilize_block(input_cb, width_in_tiles, output_cb);
            cb_push_back(output_cb, width_in_tiles);
            cb_pop_front(input_cb, width_in_tiles);
        }

        if constexpr (do_uninit) {
            ::untilize_uninit(input_cb);
        }

    } else if constexpr (width_in_tiles > dest_limit && is_integer) {
        // Block-based pack untilize (wide integer types)
        constexpr uint32_t num_blocks = compute_num_blocks(width_in_tiles, dest_limit);
        constexpr uint32_t block_w = width_in_tiles / num_blocks;

        if constexpr (do_init) {
            pack_untilize_init<block_w, width_in_tiles>(input_cb, output_cb);
        }

        for (uint32_t r = 0; r < num_rows; ++r) {
            cb_reserve_back(output_cb, width_in_tiles);
            for (uint32_t b = 0; b < num_blocks; ++b) {
                cb_wait_front(input_cb, block_w);
                pack_untilize_block<block_w, width_in_tiles>(input_cb, 1, output_cb, b);
                cb_pop_front(input_cb, block_w);
            }
            cb_push_back(output_cb, width_in_tiles);
        }

        if constexpr (do_uninit) {
            pack_untilize_uninit(output_cb);
        }

    } else {
        // Pack untilize path (width fits in DEST)
        if constexpr (do_init) {
            pack_untilize_init<width_in_tiles, width_in_tiles>(input_cb, output_cb);
        }

        const uint32_t tiles_per_iteration = width_in_tiles * row_height_tiles;

        for (uint32_t r = 0; r < num_rows; ++r) {
            cb_wait_front(input_cb, tiles_per_iteration);
            cb_reserve_back(output_cb, tiles_per_iteration);
            pack_untilize_block<width_in_tiles, width_in_tiles>(input_cb, row_height_tiles, output_cb, 0);
            cb_pop_front(input_cb, tiles_per_iteration);
            cb_push_back(output_cb, tiles_per_iteration);
        }

        if constexpr (do_uninit) {
            pack_untilize_uninit(output_cb);
        }
    }
}

}  // namespace compute_kernel_lib

// Make config types available without namespace prefix when header is included
using compute_kernel_lib::InputCB;
using compute_kernel_lib::OutputCB;
using compute_kernel_lib::UntilizeConfig;
using compute_kernel_lib::UntilizeFlags;
using compute_kernel_lib::WidthInTiles;
