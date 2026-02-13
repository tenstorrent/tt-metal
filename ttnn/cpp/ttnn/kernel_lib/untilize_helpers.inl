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

#include "ttnn/cpp/ttnn/kernel_lib/cb_helpers.hpp"

namespace compute_kernel_lib {

// =============================================================================
// Block Splitting Helper for Wide Untilize
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
    constexpr bool use_block_based_pack = (block_width_tiles > dest_limit);
    constexpr uint32_t num_sub_blocks = use_block_based_pack ? compute_num_blocks(block_width_tiles, dest_limit) : 1;
    constexpr uint32_t sub_block_width = use_block_based_pack ? (block_width_tiles / num_sub_blocks) : block_width_tiles;

    if constexpr (use_block_based_pack) {
        pack_untilize_init<sub_block_width, block_width_tiles>(input_cb, output_cb);
    } else {
        pack_untilize_init<block_width_tiles, block_width_tiles>(input_cb, output_cb);
    }
}

template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb>
ALWI void untilize_uninit() {
    pack_untilize_uninit(output_cb);
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
    static_assert(input_cb < NUM_CIRCULAR_BUFFERS,
        "Invalid input_cb: must be less than NUM_CIRCULAR_BUFFERS");
    static_assert(output_cb < NUM_CIRCULAR_BUFFERS,
        "Invalid output_cb: must be less than NUM_CIRCULAR_BUFFERS");

    // Runtime parameter validation
    ASSERT(num_blocks > 0);

    // Validate CB page sizes match expected tile sizes
    UNPACK(ASSERT(is_valid_cb_tile_page_size(input_cb, (DataFormat)unpack_src_format[input_cb])));
    PACK(ASSERT(is_valid_cb_tile_page_size(output_cb, (DataFormat)pack_dst_format[output_cb])));

    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;

    // Determine if we're doing data type reconfiguration
    constexpr bool use_unpack_reconfig =
        (reconfig_mode == untilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure) ||
        (reconfig_mode == untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    constexpr bool use_pack_reconfig =
        (reconfig_mode == untilize_config::ReconfigureRegisterDatatypeMode::PackReconfigure) ||
        (reconfig_mode == untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    // Reconfigure register datatypes if requested
    if constexpr (use_unpack_reconfig) {
        reconfig_data_format_srca(input_cb);
    }

    if constexpr (use_pack_reconfig) {
        pack_reconfig_data_format(output_cb);
    }

    // Determine which dispatch path to use
    constexpr bool use_block_based_pack = (block_width_tiles > dest_limit);

    // Compute block parameters for block-based pack path
    constexpr uint32_t num_sub_blocks = use_block_based_pack ?
        compute_num_blocks(block_width_tiles, dest_limit) : 1;
    constexpr uint32_t sub_block_width = use_block_based_pack ?
        (block_width_tiles / num_sub_blocks) : block_width_tiles;

    // Validate CB capacity
    ASSERT(get_cb_num_pages(output_cb) >= block_width_tiles);
    if constexpr (wait_mode == untilize_config::WaitMode::WaitUpfront) {
        ASSERT(get_cb_num_pages(input_cb) >= block_width_tiles * num_blocks);
    } else if constexpr (wait_mode == untilize_config::WaitMode::WaitBlock) {
        if constexpr (use_block_based_pack) {
            ASSERT(get_cb_num_pages(input_cb) >= sub_block_width);
        } else {
            ASSERT(get_cb_num_pages(input_cb) >= block_width_tiles);
        }
    }

    // =================================================================
    // INITIALIZATION
    // =================================================================

    if constexpr (
        init_uninit_mode == untilize_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == untilize_config::InitUninitMode::InitOnly) {

        if constexpr (use_block_based_pack) {
            pack_untilize_init<sub_block_width, block_width_tiles>(input_cb, output_cb);
        } else {
            pack_untilize_init<block_width_tiles, block_width_tiles>(input_cb, output_cb);
        }
    }

    // =================================================================
    // UPFRONT WAITING (if requested)
    // =================================================================

    if constexpr (wait_mode == untilize_config::WaitMode::WaitUpfront) {
        uint32_t total_tiles = block_width_tiles * num_blocks;
        cb_wait_front(input_cb, total_tiles);
    }

    // =================================================================
    // MAIN PROCESSING LOOP
    // =================================================================

    if constexpr (use_block_based_pack) {
        // =============================================================
        // BLOCK-BASED PACK UNTILIZE PATH
        // Used when width exceeds DEST limit
        // Splits wide rows into multiple sub-blocks that each fit in DEST
        // =============================================================

        for (uint32_t r = 0; r < num_blocks; ++r) {
            cb_reserve_back(output_cb, block_width_tiles);
            for (uint32_t b = 0; b < num_sub_blocks; ++b) {
                if constexpr (wait_mode == untilize_config::WaitMode::WaitBlock) {
                    cb_wait_front(input_cb, sub_block_width);
                }
                pack_untilize_block<sub_block_width, block_width_tiles>(input_cb, 1, output_cb, b);
                cb_pop_front(input_cb, sub_block_width);
            }
            cb_push_back(output_cb, block_width_tiles);
        }

    } else {
        // =============================================================
        // PACK UNTILIZE PATH (SINGLE-PASS)
        // Used when width fits in DEST (optimal path)
        // =============================================================

        for (uint32_t r = 0; r < num_blocks; ++r) {
            if constexpr (wait_mode == untilize_config::WaitMode::WaitBlock) {
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

        pack_untilize_uninit(output_cb);
    }
}

}  // namespace compute_kernel_lib
