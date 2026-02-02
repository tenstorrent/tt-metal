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

namespace compute_kernel_lib {

template <
    uint32_t input_cb,
    uint32_t output_cb,
    tilize_config::InitUninitMode init_uninit_mode,
    tilize_config::WaitMode wait_mode,
    tilize_config::TilizeSpeedMode speed_mode,
    uint32_t reconfig_from_cb>
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
    static_assert(
        (reconfig_from_cb == tilize_config::INVALID_CB) ||
        init_uninit_mode != tilize_config::InitUninitMode::Neither,
        "Data type reconfiguration requires either init or uninit: "
        "cannot use InitUninitMode::Neither with reconfig_from_cb");

    // Determine if we're using fast tilize mode (explicit, NOT auto-detected)
    constexpr bool use_fast = (speed_mode == tilize_config::TilizeSpeedMode::Fast);

    // Determine if we're doing data type reconfiguration
    constexpr bool use_dt = (reconfig_from_cb != tilize_config::INVALID_CB);

    // Compile-time initialization based on InitUninitMode
    if constexpr (
        init_uninit_mode == tilize_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == tilize_config::InitUninitMode::InitOnly) {

        if constexpr (use_dt && use_fast) {
            // Fast data type reconfiguration mode
            fast_tilize_init_with_dt(input_cb, block_width_tiles, output_cb);
        } else if constexpr (use_dt) {
            // Standard data type reconfiguration mode
            tilize_init_short_with_dt(reconfig_from_cb, input_cb, block_width_tiles, output_cb);
        } else if constexpr (use_fast) {
            // Fast tilize mode (no DT reconfiguration)
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
            // Fast tilize mode (works with both DT and non-DT)
            fast_tilize_uninit(input_cb, output_cb);
        } else if constexpr (use_dt) {
            // Standard data type reconfiguration mode
            tilize_uninit_with_dt(input_cb, reconfig_from_cb, output_cb);
        } else {
            // Standard tilize mode
            tilize_uninit(input_cb, output_cb);
        }
    }
}

}  // namespace compute_kernel_lib
