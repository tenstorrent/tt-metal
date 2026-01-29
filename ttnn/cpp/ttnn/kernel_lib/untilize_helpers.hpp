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
 * - Small widths (<= DEST limit): Uses pack_untilize (hardware-accelerated, preferred)
 * - Large widths (> DEST limit) with non-fp32 types: Uses block-based pack_untilize (hardware-accelerated)
 * - Large widths (> DEST limit) with fp32 types: Uses standard untilize (fallback)
 * - WaitUpfront mode: Uses standard untilize (pack_untilize doesn't support this pattern)
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
 *   // Simple usage - defaults work
 *   compute_kernel_lib::untilize<tiles, cb_in, cb_out>(num_blocks);
 *
 *   // Wait upfront pattern (GroupNorm)
 *   compute_kernel_lib::untilize<per_core_N, cb_in, cb_out,
 *       InitUninitMode::InitAndUninit, WaitMode::WaitUpfront>(per_core_M);
 *
 *   // No init/uninit in loop
 *   compute_kernel_lib::untilize<Wt, cb_in, cb_out,
 *       InitUninitMode::Neither>(1);
 */

namespace compute_kernel_lib {

// =============================================================================
// Constants
// =============================================================================

/// Invalid CB sentinel value (matches NUM_CIRCULAR_BUFFERS)
/// Used to indicate no DT reconfiguration when passed as reconfig_from_cb
constexpr uint32_t INVALID_CB = 32;

// get_dest_limit() and DEST_AUTO_LIMIT are provided by dest_helpers.hpp

// =============================================================================
// Enums (matching tilize pattern)
// =============================================================================

/**
 * @brief Controls init/uninit behavior at function boundaries
 *
 * InitAndUninit: Default - standalone operation, calls both init and uninit
 * InitOnly: First in a sequence of untilize operations, calls only init
 * UninitOnly: Last in a sequence, calls only uninit
 * Neither: Middle of a sequence, skips both init and uninit
 */
enum class InitUninitMode : uint8_t { InitAndUninit, InitOnly, UninitOnly, Neither };

/**
 * @brief Controls whether and when the function waits for input data
 *
 * Wait: Default - calls cb_wait_front for block_width_tiles per iteration
 * WaitUpfront: Wait for all tiles before processing starts (GroupNorm pattern)
 * NoWait: No waiting - caller manages synchronization
 */
enum class WaitMode : uint8_t { Wait, WaitUpfront, NoWait };

// =============================================================================
// Internal Helpers (declarations)
// =============================================================================

// unpack_dst_format is defined in JIT-generated chlkc_unpack_data_format.h
// It's an array where unpack_dst_format[cb_id] contains the DataFormat enum value

// Integer data formats from tt_metal/hw/inc/tt-1xx/blackhole/tensix_types.h:
// - Int8 = 14, UInt8 = 30, UInt16 = 9, Int32 = 8, UInt32 = 24

// FP32 data formats: Float32 = 4, TF32 = 20

/**
 * @brief Check if data format is an integer type
 * @tparam cb_id Circular buffer ID to check
 * @return true if CB has an integer format
 */
template <uint32_t cb_id>
constexpr bool is_integer_format();

/**
 * @brief Check if data format is FP32 (requires standard untilize for wide widths)
 *
 * FP32 formats need special handling because pack_untilize block-based path
 * doesn't work correctly for FP32 with wide widths.
 *
 * @tparam cb_id Circular buffer ID to check
 * @return true if CB has FP32 or TF32 format
 */
template <uint32_t cb_id>
constexpr bool is_fp32_format();

/**
 * @brief Compute number of column chunks needed to split a wide row into DEST-sized chunks
 *
 * Finds the largest divisor of total_width that is <= max_block_width
 * This ensures optimal block size while respecting DEST register limits.
 *
 * @param total_width Total width in tiles to be split
 * @param max_block_width Maximum block width (DEST register limit)
 * @return Number of column chunks needed
 */
constexpr uint32_t compute_num_columns(uint32_t total_width, uint32_t max_block_width);

// =============================================================================
// Unified Init/Uninit Functions (declarations)
// =============================================================================

/**
 * @brief Initialize untilize - automatically dispatches based on width and data format
 *
 * @tparam block_width_tiles Width in tiles
 * @tparam input_cb Input circular buffer ID
 * @tparam output_cb Output circular buffer ID (only used for pack path)
 */
template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb = 0>
ALWI void untilize_init();

/**
 * @brief Uninitialize untilize - automatically dispatches based on width and data format
 *
 * @tparam block_width_tiles Width in tiles
 * @tparam input_cb Input circular buffer ID (only used for standard path)
 * @tparam output_cb Output circular buffer ID (only used for pack path)
 */
template <uint32_t block_width_tiles, uint32_t input_cb = 0, uint32_t output_cb = 0>
ALWI void untilize_uninit();

// =============================================================================
// Main Function (declaration)
// =============================================================================

/**
 * @brief Unified untilize function - automatically dispatches based on width, data format, and pattern
 *
 * This is the ONLY untilize function you need. Provide the tile width and CB IDs,
 * and the optimal implementation is selected at compile time based on:
 * 1. Auto-detected DEST register capacity
 * 2. Auto-detected data format (FP32 vs non-FP32)
 * 3. Wait mode (WaitUpfront requires standard path)
 *
 * Dispatch logic:
 * - wait_mode == WaitUpfront: Always standard untilize (pack_untilize doesn't support)
 * - block_width_tiles > DEST capacity AND FP32: Standard untilize (fallback for FP32)
 * - block_width_tiles > DEST capacity AND non-FP32: Block-based pack_untilize (hardware-accelerated)
 * - block_width_tiles <= DEST capacity: Pack untilize (hardware-accelerated, single-pass)
 *
 * IMPORTANT: Requires compute kernel hardware initialization.
 * Call compute_kernel_hw_startup() before using.
 *
 * @tparam block_width_tiles Width in tiles (number of tiles per row)
 * @tparam input_cb Input circular buffer ID (tiled data) - must be compile-time constant
 * @tparam output_cb Output circular buffer ID (row-major data) - must be compile-time constant
 * @tparam init_uninit_mode Controls init/uninit behavior (default: InitAndUninit)
 * @tparam wait_mode Controls when/whether to wait for input (default: Wait)
 * @tparam reconfig_from_cb CB to reconfigure datatype from (default: INVALID_CB = disabled)
 *
 * @param num_blocks Number of blocks (rows) to process
 *
 * @example
 *   // Simple usage - defaults work
 *   untilize<tiles, cb_in, cb_out>(num_blocks);
 *
 * @example
 *   // Wait upfront pattern (GroupNorm)
 *   untilize<per_core_N, cb_in, cb_out,
 *       InitUninitMode::InitAndUninit, WaitMode::WaitUpfront>(per_core_M);
 *
 * @example
 *   // No init/uninit in loop (KV Cache)
 *   untilize<Wt, cb_in, cb_out, InitUninitMode::Neither>(1);
 *
 * @example
 *   // Sliding window halo (multiple rows with WaitUpfront)
 *   untilize<tiles_per_row, src_cb, out_cb,
 *       InitUninitMode::Neither, WaitMode::WaitUpfront>(block_size);
 */
template <
    uint32_t block_width_tiles,
    uint32_t input_cb,
    uint32_t output_cb,
    InitUninitMode init_uninit_mode = InitUninitMode::InitAndUninit,
    WaitMode wait_mode = WaitMode::Wait,
    uint32_t reconfig_from_cb = INVALID_CB>
ALWI void untilize(uint32_t num_blocks);

}  // namespace compute_kernel_lib

// Include implementation
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl"
