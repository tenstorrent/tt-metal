// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file tilize_helpers.inl
 * @brief Implementation of tilize helper functions
 *
 * This file contains the implementation details for the tilize() function.
 * It should only be included by tilize_helpers.hpp.
 */
#if __has_include("chlkc_pack_tile_dims.h")
#include "chlkc_pack_tile_dims.h"
#define PACK_TILE_DIMS_AVAILABLE
#endif
namespace compute_kernel_lib {

// =============================================================================
// Internal Helper Implementations
// =============================================================================

template <uint32_t cb_id>
constexpr bool has_32x32_tiles() {
#ifdef PACK_TILE_DIMS_AVAILABLE
    // Access pack tile dimensions at compile time
    constexpr uint32_t tile_r_dim = pack_tile_r_dim[cb_id];
    constexpr uint32_t tile_c_dim = pack_tile_c_dim[cb_id];

    // Fast tilize requires 32x32 tiles
    return tile_r_dim == 32 && tile_c_dim == 32;
#else
    // If header not available, assume 32x32 tiles (conservative)
    // fast_tilize already falls back to standard tilize on Blackhole
    return true;
#endif
}

template <uint32_t output_cb>
constexpr bool can_use_fast_tilize() {
    return has_32x32_tiles<output_cb>() && !get_dst_full_sync_enabled();
}

// =============================================================================
// Main Function Implementation
// =============================================================================

template <
    uint32_t input_cb,
    uint32_t output_cb,
    tilize_config::InitUninitMode init_uninit_mode,
    tilize_config::WaitMode wait_mode,
    tilize_config::ReconfigureRegisterDatatypeMode reconfig_mode>
ALWI void tilize(
    uint32_t block_width_tiles,
    uint32_t num_blocks,
    tilize_config::NonTileAlignedCBWaitConfig config) {

    // Compile-time validation
    static_assert(input_cb != output_cb,
        "Tilize cannot be done in-place: input_cb and output_cb must be different");
    static_assert(input_cb < 32,
        "Invalid input_cb: must be less than 32");
    static_assert(output_cb < 32,
        "Invalid output_cb: must be less than 32");

    // Runtime parameter validation
    ASSERT(block_width_tiles > 0);
    ASSERT(num_blocks > 0);

    // Determine if we're using fast tilize mode (explicit, NOT auto-detected)
    constexpr bool use_fast = can_use_fast_tilize<output_cb>();

    // Determine if we're doing data type reconfiguration
    constexpr bool use_unpack_reconfig =
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure) ||
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    constexpr bool use_pack_reconfig =
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::PackReconfigure) ||
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    // Validate NonTileAlignedCBWaitConfig parameters
    if (config.mode != tilize_config::NonTileAlignedMode::Disabled) {
        ASSERT(config.value > 0);
    }

    // TODO don't wait more than buffer size

    // Validate input CB page size for standard tile-aligned mode
    if (config.mode == tilize_config::NonTileAlignedMode::Disabled) {
        UNPACK({
            uint32_t operand_id = get_operand_id(input_cb);
            uint32_t input_page_size_units = get_local_cb_interface(operand_id).fifo_page_size;
            // fifo_page_size is in 16-byte units, convert to actual bytes
            uint32_t input_page_size = input_page_size_units << 4;
            // uint8, uint16,bfp16, uint32,fp32 -> don't hardcode values
            ASSERT(input_page_size == 1024 || input_page_size == 2048 || input_page_size == 4096);
        })
    }

    // Reconfigure register datatypes if requested
    if constexpr (use_unpack_reconfig) {
        // Reconfigure srcA for unpack
        reconfig_data_format_srca(input_cb);

        if constexpr (use_fast) {
            // Reconfigure srcB only in fast mode
            reconfig_data_format_srcb(input_cb);
        }
    }

    if constexpr (use_pack_reconfig) {
        // Reconfigure output for pack
        pack_reconfig_data_format(output_cb);
    }

    // Compile-time initialization based on InitUninitMode
    if constexpr (
        init_uninit_mode == tilize_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == tilize_config::InitUninitMode::InitOnly) {

        if constexpr (use_fast) {
            // Fast tilize mode
            fast_tilize_init(input_cb, block_width_tiles, output_cb);
        } else {
            // Standard tilize mode
            tilize_init(input_cb, block_width_tiles, output_cb);
        }
    }

    // Main processing loop - handles three patterns based on NonTileAlignedCBWaitConfig:
    // 1. TotalBatched: Variable row alignment (Conv3D pattern) - chunks of 32
    // 2. PerIteration: Asymmetric input/output (HWC/SSM pattern) - custom per-iteration count
    // 3. Disabled: Standard tile-aligned (default) - symmetric block_width_tiles

    if (config.mode == tilize_config::NonTileAlignedMode::TotalBatched) {
        // Variable row alignment pattern (Conv3D)
        // Handles non-tile-aligned input where the last iteration may have fewer rows
        // Processes in chunks of TILE_HEIGHT (32) rows
        uint32_t rows_left = config.value;  // total_pages
        constexpr uint32_t TILE_HEIGHT = 32;  // Standard tile height for all architectures

        for (uint32_t block = 0; block < num_blocks; ++block) {
            // Calculate current input rows (min of rows_left or TILE_HEIGHT)
            uint32_t current_input = (rows_left < TILE_HEIGHT) ? rows_left : TILE_HEIGHT;

            // Handle waiting based on WaitMode
            if constexpr (wait_mode == tilize_config::WaitMode::WaitBlock) {
                cb_wait_front(input_cb, current_input);
            } else if constexpr (wait_mode == tilize_config::WaitMode::WaitUpfront) {
                // WaitUpfront: wait for all data at the beginning
                if (block == 0) {
                    cb_wait_front(input_cb, config.value);  // wait for all total_pages
                }
            }
            // NoWait: skip cb_wait_front entirely

            cb_reserve_back(output_cb, block_width_tiles);

            // Compile-time selection of tilize function
            if constexpr (use_fast) {
                fast_tilize_block(input_cb, block_width_tiles, output_cb);
            } else {
                tilize_block(input_cb, block_width_tiles, output_cb);
            }

            cb_push_back(output_cb, block_width_tiles);
            cb_pop_front(input_cb, current_input);

            rows_left -= current_input;
        }

    } else {
        // Standard or asymmetric pattern
        // Determine input wait/pop count
        uint32_t input_amount;
        if (config.mode == tilize_config::NonTileAlignedMode::PerIteration) {
            // Asymmetric: custom pages per iteration
            input_amount = config.value;  // pages_per_iteration
        } else {
            // Standard tile-aligned: use block_width_tiles
            input_amount = block_width_tiles;
        }

        // Handle upfront waiting if requested
        if constexpr (wait_mode == tilize_config::WaitMode::WaitUpfront) {
            uint32_t total_tiles = block_width_tiles * num_blocks;
            cb_wait_front(input_cb, total_tiles);
        }

        for (uint32_t block = 0; block < num_blocks; ++block) {
            // Handle per-iteration waiting
            if constexpr (wait_mode == tilize_config::WaitMode::WaitBlock) {
                cb_wait_front(input_cb, input_amount);
            }
            // WaitUpfront: already waited above
            // NoWait: skip cb_wait_front

            cb_reserve_back(output_cb, block_width_tiles);

            // Compile-time selection of tilize function
            if constexpr (use_fast) {
                fast_tilize_block(input_cb, block_width_tiles, output_cb);
            } else {
                tilize_block(input_cb, block_width_tiles, output_cb);
            }

            cb_push_back(output_cb, block_width_tiles);
            cb_pop_front(input_cb, input_amount);
        }
    }

    // Compile-time cleanup based on InitUninitMode
    if constexpr (
        init_uninit_mode == tilize_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == tilize_config::InitUninitMode::UninitOnly) {

        if constexpr (use_fast) {
            // Fast tilize mode
            fast_tilize_uninit(input_cb, output_cb);
        } else {
            // Standard tilize mode
            tilize_uninit(input_cb, output_cb);
        }
    }
}

}  // namespace compute_kernel_lib
