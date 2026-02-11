// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file untilize_helpers.inl
 * @brief Implementation of untilize helper functions
 *
 * This file contains the implementation details for the untilize() function.
 * It should only be included by untilize_helpers.hpp.
 */

namespace compute_kernel_lib {

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
// If header not available, assume non integer for wide widths
// This ensures wide integer tensors get hardware acceleration
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
// Standalone Init/Uninit Wrapper Functions Implementations
// =============================================================================

template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb>
ALWI void untilize_init() {
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool is_integer = is_integer_format<input_cb>();
    constexpr bool use_standard_path = (block_width_tiles > dest_limit && !is_integer);
    constexpr uint32_t num_sub_blocks = use_standard_path ? 1 : compute_num_blocks(block_width_tiles, dest_limit);
    constexpr uint32_t sub_block_width = use_standard_path ? block_width_tiles : (block_width_tiles / num_sub_blocks);

    if constexpr (use_standard_path) {
        ::untilize_init(input_cb);
    } else if constexpr (num_sub_blocks > 1) {
        pack_untilize_init<sub_block_width, block_width_tiles>(input_cb, output_cb);
    } else {
        pack_untilize_init<block_width_tiles, block_width_tiles>(input_cb, output_cb);
    }
}

template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb>
ALWI void untilize_uninit() {
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool is_integer = is_integer_format<input_cb>();
    constexpr bool use_standard_path = (block_width_tiles > dest_limit && !is_integer);

    if constexpr (use_standard_path) {
        ::untilize_uninit(input_cb);
    } else {
        pack_untilize_uninit(output_cb);
    }
}

// =============================================================================
// Main Untilize Function Implementation
// =============================================================================

template <
    uint32_t block_width_tiles,
    uint32_t input_cb,
    uint32_t output_cb,
    untilize_config::InitUninitMode init_uninit_mode,
    untilize_config::WaitMode wait_mode,
    untilize_config::ReconfigureRegisterDatatypeMode reconfig_mode>
ALWI void untilize(uint32_t num_blocks) {

    // Compile-time validation
    static_assert(input_cb != output_cb,
        "Untilize cannot be done in-place: input_cb and output_cb must be different");
    static_assert(block_width_tiles > 0,
        "block_width_tiles must be greater than 0");
    static_assert(input_cb < 32,
        "Invalid input_cb: must be less than 32");
    static_assert(output_cb < 32,
        "Invalid output_cb: must be less than 32");

    // Runtime parameter validation
    ASSERT(num_blocks > 0);

    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool is_integer = is_integer_format<input_cb>();

    // Determine if we're doing data type reconfiguration
    constexpr bool use_unpack_reconfig =
        (reconfig_mode == untilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure) ||
        (reconfig_mode == untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    constexpr bool use_pack_reconfig =
        (reconfig_mode == untilize_config::ReconfigureRegisterDatatypeMode::PackReconfigure) ||
        (reconfig_mode == untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    // Reconfigure register datatypes if requested
    if constexpr (use_unpack_reconfig) {
        // Reconfigure srcA for unpack
        reconfig_data_format_srca(input_cb);
    }

    if constexpr (use_pack_reconfig) {
        // Reconfigure output for pack
        pack_reconfig_data_format(output_cb);
    }

    // Determine which dispatch path to use
    // WaitUpfront or non-integer wide widths always use standard path
    constexpr bool use_standard_path =
        (wait_mode == untilize_config::WaitMode::WaitUpfront) ||
        (block_width_tiles > dest_limit && !is_integer);

    constexpr bool use_block_based_pack =
        (block_width_tiles > dest_limit && is_integer &&
         wait_mode != untilize_config::WaitMode::WaitUpfront);

    // Compute block parameters for block-based pack path
    constexpr uint32_t num_sub_blocks = use_block_based_pack ?
        compute_num_blocks(block_width_tiles, dest_limit) : 1;
    constexpr uint32_t sub_block_width = use_block_based_pack ?
        (block_width_tiles / num_sub_blocks) : block_width_tiles;

    // =================================================================
    // INITIALIZATION
    // =================================================================

    if constexpr (
        init_uninit_mode == untilize_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == untilize_config::InitUninitMode::InitOnly) {

        if constexpr (use_standard_path) {
            // Standard untilize path initialization
            ::untilize_init(input_cb);
        } else if constexpr (use_block_based_pack) {
            // Block-based pack untilize initialization
            pack_untilize_init<sub_block_width, block_width_tiles>(input_cb, output_cb);
        } else {
            // Single-pass pack untilize initialization
            pack_untilize_init<block_width_tiles, block_width_tiles>(input_cb, output_cb);
        }
    }

    // =================================================================
    // MAIN PROCESSING LOOP
    // =================================================================

    if constexpr (use_standard_path) {
        // =================================================================
        // STANDARD UNTILIZE PATH
        // Used when:
        // - WaitMode::WaitUpfront (GroupNorm pattern)
        // - Width exceeds DEST AND non-integer type (float fallback)
        // =================================================================

        // Handle upfront waiting
        if constexpr (wait_mode == untilize_config::WaitMode::WaitUpfront) {
            uint32_t total_tiles = block_width_tiles * num_blocks;
            cb_wait_front(input_cb, total_tiles);
        }

        for (uint32_t r = 0; r < num_blocks; ++r) {
            // Handle per-row waiting
            if constexpr (wait_mode == untilize_config::WaitMode::WaitBlock) {
                cb_wait_front(input_cb, block_width_tiles);
            }
            // WaitUpfront: already waited above
            // NoWait: skip cb_wait_front

            cb_reserve_back(output_cb, block_width_tiles);
            untilize_block(input_cb, block_width_tiles, output_cb);
            cb_push_back(output_cb, block_width_tiles);
            cb_pop_front(input_cb, block_width_tiles);
        }

    } else if constexpr (use_block_based_pack) {
        // =================================================================
        // BLOCK-BASED PACK UNTILIZE PATH
        // Used for integer types with width exceeding DEST limit
        // Splits wide rows into multiple sub-blocks that each fit in DEST
        // Provides hardware acceleration for wide integer tensors
        // =================================================================

        for (uint32_t r = 0; r < num_blocks; ++r) {
            cb_reserve_back(output_cb, block_width_tiles);
            for (uint32_t b = 0; b < num_sub_blocks; ++b) {
                if constexpr (wait_mode != untilize_config::WaitMode::NoWait) {
                    cb_wait_front(input_cb, sub_block_width);
                }
                pack_untilize_block<sub_block_width, block_width_tiles>(input_cb, 1, output_cb, b);
                cb_pop_front(input_cb, sub_block_width);
            }
            cb_push_back(output_cb, block_width_tiles);
        }

    } else {
        // =================================================================
        // PACK UNTILIZE PATH (SINGLE-PASS)
        // Used when width fits in DEST (optimal for all data types)
        // =================================================================

        for (uint32_t r = 0; r < num_blocks; ++r) {
            if constexpr (wait_mode != untilize_config::WaitMode::NoWait) {
                cb_wait_front(input_cb, block_width_tiles);
            }
            cb_reserve_back(output_cb, block_width_tiles);
            pack_untilize_block<block_width_tiles, block_width_tiles>(input_cb, 1, output_cb, 0);
            cb_pop_front(input_cb, block_width_tiles);
            cb_push_back(output_cb, block_width_tiles);
        }
    }

    // =================================================================
    // CLEANUP
    // =================================================================

    if constexpr (
        init_uninit_mode == untilize_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == untilize_config::InitUninitMode::UninitOnly) {

        if constexpr (use_standard_path) {
            // Standard untilize path cleanup
            ::untilize_uninit(input_cb);
        } else {
            // Pack untilize path cleanup (both single-pass and block-based)
            pack_untilize_uninit(output_cb);
        }
    }
}

}  // namespace compute_kernel_lib
