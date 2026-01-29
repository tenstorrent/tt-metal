// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

// Common types shared with tilize_helpers.hpp:
//   - INVALID_CB: sentinel value (32) indicating no CB for reconfig
//   - InitUninitMode: { InitAndUninit, InitOnly, UninitOnly, Neither }
//   - WaitMode: { Wait, WaitUpfront, NoWait }
#include "ttnn/cpp/ttnn/kernel_lib/compute_kernel_lib_common.hpp"

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

// INVALID_CB, InitUninitMode, and WaitMode are provided by compute_kernel_lib_common.hpp
// get_dest_limit() and DEST_AUTO_LIMIT are provided by dest_helpers.hpp

// =============================================================================
// Unified Init/Uninit Functions (declarations)
// =============================================================================

/**
 * @brief Initialize untilize - automatically dispatches based on width and data format
 *
 * @tparam block_width_tiles Width in tiles
 * @tparam input_cb Input circular buffer ID
 * @tparam output_cb Output circular buffer ID
 */
template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb>
ALWI void untilize_init();

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
