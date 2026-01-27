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
 * - Large widths (> DEST limit) with Float32/integer types: Uses block-based pack_untilize (hardware-accelerated)
 * - Large widths (> DEST limit) with other formats (bfloat16, etc.): Uses standard untilize (fallback)
 *
 * Float32 and integer types require block-based pack_untilize because standard untilize
 * has correctness issues with these formats (see TODO #30400, #33795).
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

// Data formats from tt_metal/hw/inc/tt-1xx/blackhole/tensix_types.h:
// - Float32 = 0
// - Int8 = 14
// - UInt8 = 30
// - UInt16 = 9
// - Int32 = 8
// - UInt32 = 24

/**
 * @brief Check if format requires block-based pack_untilize for wide tensors
 *
 * Some data formats cannot use standard untilize when width exceeds DEST limit:
 * - Integer types: standard untilize doesn't support them
 * - Float32: standard untilize has precision/correctness issues (see TODO #30400, #33795)
 *
 * These formats must use block-based pack_untilize which chunks the row into
 * DEST-sized blocks.
 *
 * @tparam cb_id The circular buffer ID
 * @tparam force_pack_fp32 If true, Float32 will use block-based pack_untilize (default: true)
 */
template <uint32_t cb_id, bool force_pack_fp32 = true>
constexpr bool requires_block_pack_untilize_for_wide() {
// Check if unpack_dst_format array is available (from JIT-generated chlkc_unpack_data_format.h)
// This header is included via chlkc_list.h in the firmware build
#if __has_include("chlkc_unpack_data_format.h")
// Include the JIT-generated header
#include "chlkc_unpack_data_format.h"

    // Access the format at compile time
    constexpr uint32_t format = unpack_dst_format[cb_id];

    // Formats that require block-based pack_untilize when width exceeds DEST limit:
    // Integer types always require it, Float32 is controlled by force_pack_fp32 parameter
    constexpr bool is_integer = format == 8 ||   // Int32
                                format == 24 ||  // UInt32
                                format == 14 ||  // Int8
                                format == 30 ||  // UInt8
                                format == 9;     // UInt16
    constexpr bool is_fp32 = (format == 0);      // Float32

    return is_integer || (is_fp32 && force_pack_fp32);
#else
    // If header not available, assume NOT requiring block pack (conservative)
    // This ensures formats like bfloat16/bfloat8_b use the standard untilize path
    // for wide tensors, which works correctly for those formats.
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
    constexpr UntilizeFlags flags = Config::flags;
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool force_pack_fp32 = has_flag(flags, UntilizeFlags::FORCE_PACK_UNTILIZE_WIDE_FP32);
    constexpr bool use_block_pack = requires_block_pack_untilize_for_wide<input_cb, force_pack_fp32>();

    if constexpr (width_in_tiles > dest_limit && use_block_pack) {
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
    constexpr UntilizeFlags flags = Config::flags;
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool force_pack_fp32 = has_flag(flags, UntilizeFlags::FORCE_PACK_UNTILIZE_WIDE_FP32);
    constexpr bool use_block_pack = requires_block_pack_untilize_for_wide<input_cb, force_pack_fp32>();

    if constexpr (width_in_tiles > dest_limit && !use_block_pack) {
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
    constexpr bool force_pack_fp32 = has_flag(flags, UntilizeFlags::FORCE_PACK_UNTILIZE_WIDE_FP32);
    constexpr bool use_block_pack = requires_block_pack_untilize_for_wide<input_cb, force_pack_fp32>();

    if constexpr (wait_upfront || (width_in_tiles > dest_limit && !use_block_pack)) {
        // Standard untilize path (WAIT_UPFRONT or wide formats that support standard untilize)
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

    } else if constexpr (width_in_tiles > dest_limit && use_block_pack) {
        // Block-based pack untilize (wide Float32/integer types that require pack_untilize)
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
