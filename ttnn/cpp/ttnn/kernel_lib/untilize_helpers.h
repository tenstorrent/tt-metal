// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/cb_api.h"

/**
 * @file untilize_helpers.h
 * @brief Header-only kernel library for standard untilize operations
 *
 * This library provides a single unified function for standard untilize operations.
 *
 * Key Features:
 * - ONE function handles all standard untilize patterns
 * - Zero runtime overhead (all functions inlined)
 * - Template-based compile-time optimization
 * - Reduces code duplication across 16+ kernels
 *
 * IMPORTANT: Untilize functions require compute kernel hardware initialization.
 * You MUST call compute_kernel_hw_startup() before using any untilize functions.
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.h"
 *
 *   // Initialize compute kernel hardware FIRST
 *   compute_kernel_hw_startup(cb_in, cb_out);
 *
 *   // Simple loop
 *   untilize(cb_in, 32, cb_out, 10);
 *
 *   // Wait-upfront pattern
 *   untilize<true, true, true>(cb_in, 32, cb_out, 10, 320);
 */

namespace compute_kernel_lib {

/**
 * @brief Unified standard untilize function handling all patterns
 *
 * This single function handles:
 * - Simple loop (default)
 * - Wait-upfront pattern (wait_upfront = true)
 * - Optional init/uninit control
 *
 * IMPORTANT - HARDWARE INITIALIZATION REQUIREMENT:
 * Before calling this function, you MUST initialize the compute kernel hardware by
 * calling compute_kernel_hw_startup() at the start of your kernel.
 *
 * @tparam init If true, calls untilize_init before processing (default: true)
 * @tparam uninit If true, calls untilize_uninit after processing (default: true)
 *               Note: uninit is being deprecated per tt-metal#22904
 * @tparam wait_upfront If true, waits for all tiles before loop (default: false)
 *
 * @param icb Input circular buffer ID (tiled data)
 * @param block_w Block width in tiles (for untilize_block)
 * @param ocb Output circular buffer ID (row-major data)
 * @param num_blocks Number of blocks/iterations to process
 * @param total_tiles Total tiles to wait for if wait_upfront=true (default: 0)
 *                    If 0 and wait_upfront=true, calculates as block_w * num_blocks
 *
 * @example
 *   // Simple loop (most common)
 *   untilize(cb_in, 32, cb_out, 10);
 *
 * @example
 *   // Wait-upfront pattern (GroupNorm)
 *   untilize<true, true, true>(cb_in, per_core_N, cb_out, per_core_M, per_core_MN);
 *
 * @example
 *   // Skip uninit (if needed)
 *   untilize<true, false>(cb_in, block_w, cb_out, num_blocks);
 *
 * @example
 *   // Function-scoped with init/uninit (uses defaults)
 *   untilize(cb_in, tiles, cb_out, 1);
 */
template <bool init = true, bool uninit = true, bool wait_upfront = false>
ALWI void untilize(uint32_t icb, uint32_t block_w, uint32_t ocb, uint32_t num_blocks, uint32_t total_tiles = 0) {
    // Compile-time initialization
    if constexpr (init) {
        untilize_init(icb);
    }

    // Optional wait for all tiles upfront
    if constexpr (wait_upfront) {
        uint32_t wait_amount = (total_tiles > 0) ? total_tiles : (block_w * num_blocks);
        cb_wait_front(icb, wait_amount);
    }

    // Main processing loop
    for (uint32_t b = 0; b < num_blocks; ++b) {
        // Wait per iteration if not waiting upfront
        if constexpr (!wait_upfront) {
            cb_wait_front(icb, block_w);
        }

        cb_reserve_back(ocb, block_w);
        untilize_block(icb, block_w, ocb);
        cb_push_back(ocb, block_w);
        cb_pop_front(icb, block_w);
    }

    // Compile-time cleanup (being deprecated)
    if constexpr (uninit) {
        untilize_uninit(icb);
    }
}

}  // namespace compute_kernel_lib
