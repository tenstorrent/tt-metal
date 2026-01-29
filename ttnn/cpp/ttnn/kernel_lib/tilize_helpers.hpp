// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

/**
 * @file tilize_helpers.hpp
 * @brief Header-only kernel library for tilize operations
 *
 * This library provides a single unified function for ALL tilize operations.
 *
 * Key Features:
 * - ONE function handles everything
 * - Zero runtime overhead (all functions inlined)
 * - Template-based compile-time optimization
 * - Automatic fast tilize selection when requirements are met
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
 *   // Simple loop (CBs are compile-time template arguments)
 *   compute_kernel_lib::tilize<cb_in, cb_out>(32, 10);
 *
 *   // Skip init/uninit (middle of a sequence)
 *   compute_kernel_lib::tilize<cb_in, cb_out, InitUninitMode::Neither>(32, 10);
 *
 *   // Datatype reconfiguration (switch from old_cb's format)
 *   compute_kernel_lib::tilize<new_cb, cb_out, InitUninitMode::InitAndUninit,
 *                              WaitMode::Wait, old_cb>(16, 5);
 */

namespace compute_kernel_lib {

// =============================================================================
// Constants
// =============================================================================

/// Invalid CB sentinel value (matches NUM_CIRCULAR_BUFFERS)
/// Used to indicate no DT reconfiguration when passed as previous_dt_cb
constexpr uint32_t INVALID_CB = 32;

// =============================================================================
// Enums
// =============================================================================

/**
 * @brief Controls init/uninit behavior at function boundaries
 *
 * InitAndUninit: Default - standalone operation, calls both init and uninit
 * InitOnly: First in a sequence of tilize operations, calls only init
 * UninitOnly: Last in a sequence, calls only uninit
 * Neither: Middle of a sequence, skips both init and uninit
 */
enum class InitUninitMode : uint8_t { InitAndUninit, InitOnly, UninitOnly, Neither };

/**
 * @brief Controls whether the function waits for input data
 *
 * Wait: Default - calls cb_wait_front before each block iteration
 * NoWait: No waiting - caller manages synchronization (cb_wait_front called externally)
 */
enum class WaitMode : uint8_t { Wait, NoWait };

// =============================================================================
// Internal Helpers (declarations)
// =============================================================================

/**
 * @brief Check if CB has 32x32 tile dimensions at compile time
 *
 * Fast tilize requires 32x32 tile dimensions. This function checks the tile
 * configuration from JIT-generated headers when available.
 *
 * @tparam cb_id Circular buffer ID to check
 * @return true if CB has 32x32 tiles (or if unable to determine, assumes true)
 */
template <uint32_t cb_id>
constexpr bool has_32x32_tiles();

/**
 * @brief Determine if fast tilize can be used based on compile-time conditions
 *
 * Fast tilize is enabled when ALL requirements are met:
 * 1. Output CB has 32x32 tile dimensions
 * 2. Half sync mode is enabled (dst_full_sync_en=false)
 *
 * Note: Fast tilize drops FP32 precision for higher throughput.
 * On Blackhole architecture, fast_tilize internally falls back to standard tilize.
 *
 * @tparam output_cb Output circular buffer ID
 * @return true if fast tilize can be used
 */
template <uint32_t output_cb>
constexpr bool can_use_fast_tilize();

// =============================================================================
// Main Function (declaration)
// =============================================================================

/**
 * @brief Unified tilize function handling ALL patterns
 *
 * This single function handles:
 * - Simple loop
 * - Fast variants (automatically selected when requirements are met)
 * - Datatype reconfiguration (reconfig_from_cb != INVALID_CB)
 * - Page-based waiting (when input CB page size differs from tile size):
 *   - input_pages_per_block: pages to wait for per iteration
 *   - total_input_pages: total pages to wait for, chunked 32 at a time (takes priority over input_pages_per_block)
 *
 * Fast tilize is automatically enabled when:
 * 1. Output CB has 32x32 tile dimensions
 * 2. Half sync mode is enabled
 *
 * IMPORTANT - HARDWARE INITIALIZATION REQUIREMENT:
 * Before calling this function, you MUST initialize the compute kernel hardware by
 * calling compute_kernel_hw_startup() or a functional equivalent at the start of
 * your kernel. Failure to do so will result in undefined behavior.
 *
 * Example:
 *   compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
 *   tilize<tt::CBIndex::c_0, tt::CBIndex::c_16>(block_width_tiles, num_blocks);
 *
 * @tparam input_cb Input circular buffer ID (must be compile-time constant)
 * @tparam output_cb Output circular buffer ID (must be compile-time constant)
 * @tparam init_uninit_mode Controls init/uninit behavior (default: InitAndUninit)
 * @tparam wait_mode Whether to wait for input (default: Wait)
 * @tparam reconfig_from_cb CB to reconfigure datatype from (default: INVALID_CB = disabled)
 *                          When != INVALID_CB, enables datatype reconfiguration from this CB's format
 *
 * @param block_width_tiles Number of tiles per tilize call (tiles per output row)
 * @param num_blocks Number of iterations/blocks to process
 * @param input_pages_per_block Input page count to wait for per iteration (default: 0 = use block_width_tiles)
 *                              Use when input CB page size differs from tile size.
 * @param total_input_pages Total input page count to wait for across all iterations (default: 0 = disabled)
 *                          When > 0, waits are chunked 32 pages at a time (TILE_HEIGHT), with the last
 *                          iteration processing remaining pages. Takes priority over input_pages_per_block.
 *
 * @example
 *   // Simple loop
 *   tilize<cb_in, cb_out>(32, 10);
 *
 * @example
 *   // Skip init/uninit (middle of a sequence)
 *   tilize<cb_in, cb_out, InitUninitMode::Neither>(32, 10);
 *
 * @example
 *   // Data type reconfiguration
 *   tilize<new_cb, cb_out, InitUninitMode::InitAndUninit, WaitMode::Wait, old_cb>(16, 5);
 *
 * @example
 *   // Page-based with total pages (conv3d pattern) - chunked 32 at a time
 *   tilize<cb_in, cb_out>(matmul_K_t, matmul_M_t, 0, num_patches);
 *
 * @example
 *   // Page-based with pages per iteration (convert_to_hwc pattern)
 *   tilize<cb_in, cb_out>(total_tiles, 1, total_sticks);
 *
 * @example
 *   // Caller-managed waiting (pre-loaded data)
 *   tilize<cb_in, cb_out, InitUninitMode::InitAndUninit, WaitMode::NoWait>(per_core_N, per_core_M);
 */
template <
    uint32_t input_cb,
    uint32_t output_cb,
    InitUninitMode init_uninit_mode = InitUninitMode::InitAndUninit,
    WaitMode wait_mode = WaitMode::Wait,
    uint32_t reconfig_from_cb = INVALID_CB>
ALWI void tilize(
    uint32_t block_width_tiles,
    uint32_t num_blocks,
    uint32_t input_pages_per_block = 0,
    uint32_t total_input_pages = 0);

}  // namespace compute_kernel_lib

// Include implementation
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl"
