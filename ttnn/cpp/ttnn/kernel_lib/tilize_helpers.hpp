// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/cb_api.h"

/**
 * @file tilize_helpers.h
 * @brief Header-only kernel library for tilize operations
 *
 * This library provides a single unified function for ALL tilize operations.
 *
 * Key Features:
 * - ONE function handles everything
 * - Zero runtime overhead (all functions inlined)
 * - Template-based compile-time optimization
 * - Reduces code duplication across 40+ kernels
 *
 * IMPORTANT: Tilize functions require compute kernel hardware initialization.
 * You MUST call compute_kernel_hw_startup() or a functional equivalent at the
 * start of your kernel before using any tilize functions.
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
 *
 *   // Initialize compute kernel hardware FIRST
 *   compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
 *
 *   // Simple loop
 *   tilize(cb_in, 32, cb_out, 10);
 *
 *   // Activation pattern
 *   tilize(cb_in, 4, cb_out, 2, 8);
 *
 *   // Fast variant
 *   tilize<true, true, true>(cb_in, 32, cb_out, 10);
 *
 *   // Data type reconfiguration
 *   tilize<true, true, false, true>(cb_in, 16, cb_out, 5, 1, old_cb);
 */

namespace compute_kernel_lib {

/**
 * @brief Unified tilize function handling ALL patterns
 *
 * This single function handles:
 * - Simple loop (subblock_h = 1)
 * - Activation pattern (subblock_h > 1)
 * - Fast variants (use_fast = true)
 * - Data type reconfiguration (use_dt = true)
 * - Variable row alignment (total_rows > 0) - for non-tile-aligned data
 * - Asymmetric input/output counts (input_count > 0) - for different wait/pop vs reserve/push
 *
 * IMPORTANT - HARDWARE INITIALIZATION REQUIREMENT:
 * Before calling this function, you MUST initialize the compute kernel hardware by
 * calling compute_kernel_hw_startup() or a functional equivalent at the start of
 * your kernel. Failure to do so will result in undefined behavior.
 *
 * Example:
 *   compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
 *   tilize(tt::CBIndex::c_0, block_w, tt::CBIndex::c_16, num_blocks);
 *
 * @tparam init If true, calls tilize_init before processing (default: true)
 * @tparam uninit If true, calls tilize_uninit after processing (default: true)
 * @tparam use_fast If true, uses fast_tilize_* functions (default: false)
 * @tparam use_dt If true, uses DT-aware init/uninit functions (default: false)
 * @tparam skip_wait If true, skips cb_wait_front in loop (default: false)
 *                   Use when data is pre-loaded or managed externally
 *
 * @param icb Input circular buffer ID (if use_dt=true, this is the NEW CB)
 * @param block_w Block width in tiles (tiles per row for tilize_block and output reserve/push)
 * @param ocb Output circular buffer ID
 * @param num_blocks Number of blocks/subblocks to process
 * @param subblock_h Height of each subblock in tiles (default: 1 = simple loop)
 * @param old_icb Previous input CB for DT tracking (default: 0, only used if use_dt=true)
 * @param input_count Override for cb_wait_front/cb_pop_front count (default: 0 = use block_w)
 *                    Use for asymmetric patterns where input count differs from output count
 * @param total_rows Total input rows for variable row alignment (default: 0 = disabled)
 *                   When > 0, enables dynamic per-iteration row calculation for non-tile-aligned data
 *
 * @example
 *   // Simple loop
 *   tilize(cb_in, 32, cb_out, 10);
 *
 * @example
 *   // Activation pattern
 *   tilize(cb_in, 4, cb_out, 2, 8);
 *
 * @example
 *   // Fast variant
 *   tilize<true, true, true>(cb_in, 32, cb_out, 10);
 *
 * @example
 *   // Data type reconfiguration
 *   tilize<true, true, false, true>(new_cb, 16, cb_out, 5, 1, old_cb);
 *
 * @example
 *   // Fast + DT
 *   tilize<true, true, true, true>(new_cb, 64, cb_out, 5, 1, old_cb);
 *
 * @example
 *   // Variable row alignment (conv3d pattern)
 *   tilize(cb_in, matmul_K_t, cb_out, matmul_M_t, 1, 0, 0, num_patches);
 *
 * @example
 *   // Asymmetric input/output (convert_to_hwc pattern)
 *   tilize(cb_in, total_tiles, cb_out, 1, 1, 0, total_sticks);
 *
 * @example
 *   // Skip wait in loop (groupnorm pattern with pre-loaded data)
 *   tilize<true, true, false, false, true>(cb_in, per_core_N, cb_out, per_core_M);
 */
template <bool init = true, bool uninit = true, bool use_fast = false, bool use_dt = false, bool skip_wait = false>
ALWI void tilize(
    uint32_t icb,
    uint32_t block_w,
    uint32_t ocb,
    uint32_t num_blocks,
    uint32_t subblock_h = 1,
    uint32_t old_icb = 0,
    uint32_t input_count = 0,
    uint32_t total_rows = 0) {
    // Compile-time initialization
    if constexpr (init) {
        if constexpr (use_dt && use_fast) {
            // Fast data type reconfiguration mode (for conv2d)
            fast_tilize_init_with_dt(icb, block_w, ocb);
        } else if constexpr (use_dt) {
            // Standard data type reconfiguration mode
            tilize_init_short_with_dt(old_icb, icb, block_w, ocb);
        } else if constexpr (use_fast) {
            // Fast tilize mode
            fast_tilize_init(icb, block_w, ocb);
        } else {
            // Standard tilize mode
            tilize_init(icb, block_w, ocb);
        }
    }

    // Main processing loop - handles three patterns:
    // 1. Variable row alignment (total_rows > 0): Conv3D pattern
    // 2. Asymmetric input/output (input_count > 0): HWC/SSM pattern
    // 3. Standard symmetric (default): Existing behavior
    if (total_rows > 0) {
        // Variable row alignment pattern (Conv3D)
        // Handles non-tile-aligned input where the last iteration may have fewer rows
        uint32_t rows_left = total_rows;
        constexpr uint32_t TILE_HEIGHT = 32;  // Standard tile height for all architectures
        for (uint32_t block = 0; block < num_blocks; ++block) {
            for (uint32_t h = 0; h < subblock_h; ++h) {
                // Calculate current input rows (min of rows_left or TILE_HEIGHT)
                uint32_t current_input = rows_left < TILE_HEIGHT ? rows_left : TILE_HEIGHT;

                if constexpr (!skip_wait) {
                    cb_wait_front(icb, current_input);
                }
                cb_reserve_back(ocb, block_w);

                // Compile-time selection of tilize function
                if constexpr (use_fast) {
                    fast_tilize_block(icb, block_w, ocb);
                } else {
                    tilize_block(icb, block_w, ocb);
                }

                cb_push_back(ocb, block_w);
                cb_pop_front(icb, current_input);

                rows_left -= current_input;
            }
        }
    } else {
        // Standard or asymmetric pattern
        // Determine input wait/pop count (asymmetric if input_count > 0)
        uint32_t input_amount = (input_count > 0) ? input_count : block_w;

        for (uint32_t block = 0; block < num_blocks; ++block) {
            for (uint32_t h = 0; h < subblock_h; ++h) {
                if constexpr (!skip_wait) {
                    cb_wait_front(icb, input_amount);
                }
                cb_reserve_back(ocb, block_w);

                // Compile-time selection of tilize function
                if constexpr (use_fast) {
                    fast_tilize_block(icb, block_w, ocb);
                } else {
                    tilize_block(icb, block_w, ocb);
                }

                cb_push_back(ocb, block_w);
                cb_pop_front(icb, input_amount);
            }
        }
    }

    // Compile-time cleanup
    if constexpr (uninit) {
        if constexpr (use_fast) {
            // Fast tilize mode (works with both DT and non-DT)
            fast_tilize_uninit(icb, ocb);
        } else if constexpr (use_dt) {
            // Standard data type reconfiguration mode
            tilize_uninit_with_dt(icb, old_icb, ocb);
        } else {
            // Standard tilize mode
            tilize_uninit(icb, ocb);
        }
    }
}

}  // namespace compute_kernel_lib
