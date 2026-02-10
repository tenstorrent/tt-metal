// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

/**
 * @file untilize_helpers.hpp
 * @brief Single unified untilize function with improved type-safe API
 *
 * This library provides a unified untilize function with:
 * - Compile-time type safety via template parameters for CB IDs
 * - Descriptive enum-based configuration instead of boolean flags
 * - Simplified runtime parameters (only num_blocks)
 * - Automatic dispatch based on width and data format
 *
 * Dispatch logic:
 * - block_width_tiles <= DEST capacity: Pack untilize (hardware-accelerated, single-pass)
 * - block_width_tiles > DEST capacity AND integer type: Block-based pack untilize (multi-pass)
 * - block_width_tiles > DEST capacity AND non-integer: Standard untilize (fallback for floats)
 * - WaitMode::WaitUpfront: Always use standard untilize (pack_untilize doesn't support this)
 *
 * IMPORTANT: Requires compute kernel hardware initialization.
 * Call compute_kernel_hw_startup() before using.
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
 *
 *   compute_kernel_hw_startup(cb_in, cb_out);
 *
 *   // Simple untilize
 *   compute_kernel_lib::untilize<4, cb_in, cb_out>(num_rows);
 *
 *   // With wait upfront (GroupNorm pattern)
 *   using namespace compute_kernel_lib::untilize_config;
 *   compute_kernel_lib::untilize<10, cb_in, cb_out,
 *       InitUninitMode::InitAndUninit,
 *       WaitMode::WaitUpfront>(num_rows);
 */

namespace compute_kernel_lib {

// get_dest_limit() and DEST_AUTO_LIMIT are provided by dest_helpers.hpp

// Nested namespace for untilize-specific types to avoid conflicts
namespace untilize_config {

// Sentinel value for invalid/unset circular buffer ID
constexpr uint32_t INVALID_CB = NUM_CIRCULAR_BUFFERS;

/**
 * @brief Controls register datatype reconfiguration mode for untilize operations
 *
 * NoReconfigure - no register datatype reconfiguration (default)
 * UnpackReconfigure - reconfigure only unpack registers (srcA)
 * PackReconfigure - reconfigure only pack registers (output)
 * UnpackAndPackReconfigure - reconfigure both unpack and pack registers
 */
enum class ReconfigureRegisterDatatypeMode : uint8_t {
    NoReconfigure,            // No reconfiguration (default)
    UnpackReconfigure,        // Reconfigure unpack registers (srcA/srcB)
    PackReconfigure,          // Reconfigure pack registers (output)
    UnpackAndPackReconfigure  // Reconfigure both unpack and pack registers
};

/**
 * @brief Controls init/uninit behavior for untilize operations
 *
 * Use InitAndUninit for standalone operations (default).
 * Use InitOnly/UninitOnly/Neither when chaining multiple untilize calls.
 */
enum class InitUninitMode : uint8_t {
    InitAndUninit,  // Default - standalone operation (calls both init and uninit)
    InitOnly,       // First in a sequence (calls init only)
    UninitOnly,     // Last in a sequence (calls uninit only)
    Neither         // Middle of a sequence (calls neither)
};

/**
 * @brief Controls input synchronization (wait strategy) for untilize operations
 *
 * WaitBlock (default) - wait per block/row (standard behavior)
 * WaitUpfront - wait for all tiles upfront before processing (GroupNorm pattern)
 * NoWait - caller manages synchronization (currently unused, reserved for future)
 */
enum class WaitMode : uint8_t {
    WaitBlock,    // Default - wait per block/row
    WaitUpfront,  // Wait for all tiles upfront before processing
    NoWait        // Caller manages synchronization (reserved for future use)
};

}  // namespace untilize_config

// =============================================================================
// Standalone Init/Uninit Wrapper Functions
// =============================================================================

/**
 * @brief Initialize untilize operation (standalone wrapper)
 *
 * This is a convenience wrapper for manual init/uninit control.
 * Prefer using the unified untilize() function with InitUninitMode enums.
 *
 * @tparam block_width_tiles Width in tiles
 * @tparam input_cb Input circular buffer ID
 * @tparam output_cb Output circular buffer ID
 */
template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb>
ALWI void untilize_init();

/**
 * @brief Uninitialize untilize operation (standalone wrapper)
 *
 * This is a convenience wrapper for manual init/uninit control.
 * Prefer using the unified untilize() function with InitUninitMode enums.
 *
 * @tparam block_width_tiles Width in tiles
 * @tparam input_cb Input circular buffer ID
 * @tparam output_cb Output circular buffer ID
 */
template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb>
ALWI void untilize_uninit();

// =============================================================================
// Main Untilize Function
// =============================================================================

/**
 * @brief Unified untilize function with type-safe API and automatic dispatch
 *
 * This is the ONLY untilize function you need. Provide the tile width and CB IDs,
 * and the optimal implementation is selected at compile time based on:
 * 1. Auto-detected DEST register capacity
 * 2. Auto-detected data format (integer vs non-integer)
 * 3. Width constraints
 * 4. Wait mode (per-row vs upfront)
 *
 * Dispatch logic:
 * - block_width_tiles <= DEST capacity: Pack untilize (hardware-accelerated, single-pass)
 * - block_width_tiles > DEST capacity AND integer type: Block-based pack untilize (multi-pass)
 * - block_width_tiles > DEST capacity AND non-integer: Standard untilize (fallback for floats)
 * - WaitMode::WaitUpfront: Always use standard untilize (pack_untilize doesn't support this)
 *
 * Integer data types (Int8, UInt8, UInt16, Int32, UInt32) can use block-based pack_untilize
 * even for wide widths, providing hardware acceleration by splitting the row into blocks
 * that each fit within DEST register limits.
 *
 * @tparam block_width_tiles Width in tiles (number of tiles per row) - FIRST template param
 * @tparam input_cb Input circular buffer ID (tiled data) - must be compile-time constant
 * @tparam output_cb Output circular buffer ID (row-major data) - must be compile-time constant
 * @tparam init_uninit_mode Controls init/uninit behavior (default: InitAndUninit)
 * @tparam wait_mode Controls input synchronization strategy (default: Wait)
 * @tparam reconfig_mode Controls register datatype reconfiguration (default: NoReconfigure)
 *
 * @param num_blocks Number of rows/blocks to process
 *
 * @example
 *   // Simple untilize (width 4, auto-dispatches to pack_untilize)
 *   untilize<4, cb_in, cb_out>(10);
 *
 * @example
 *   // Width 32 with INT32 - automatically uses block-based pack_untilize
 *   untilize<32, cb_in, cb_out>(1);
 *
 * @example
 *   // Width 32 with Float16 - automatically uses standard untilize
 *   untilize<32, cb_in, cb_out>(10);
 *
 * @example
 *   // Wait-upfront pattern (GroupNorm) - forces standard untilize
 *   using namespace compute_kernel_lib::untilize_config;
 *   untilize<10, cb_in, cb_out,
 *            InitUninitMode::InitAndUninit,
 *            WaitMode::WaitUpfront>(num_rows);
 *
 * @example
 *   // Unpack and pack data type reconfiguration
 *   using namespace compute_kernel_lib::untilize_config;
 *   untilize<4, cb_in, cb_out,
 *            InitUninitMode::InitAndUninit,
 *            WaitMode::WaitBlock,
 *            ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(10);
 *
 * @example
 *   // Only unpack reconfiguration
 *   using namespace compute_kernel_lib::untilize_config;
 *   untilize<4, cb_in, cb_out,
 *            InitUninitMode::InitAndUninit,
 *            WaitMode::WaitBlock,
 *            ReconfigureRegisterDatatypeMode::UnpackReconfigure>(10);
 *
 * @example
 *   // Only pack reconfiguration
 *   using namespace compute_kernel_lib::untilize_config;
 *   untilize<4, cb_in, cb_out,
 *            InitUninitMode::InitAndUninit,
 *            WaitMode::WaitBlock,
 *            ReconfigureRegisterDatatypeMode::PackReconfigure>(10);
 *
 * @example
 *   // Init only (first in sequence)
 *   using namespace compute_kernel_lib::untilize_config;
 *   untilize<width, cb_in, cb_out,
 *            InitUninitMode::InitOnly>(num_blocks);
 *
 * @example
 *   // Neither init nor uninit (middle of sequence)
 *   using namespace compute_kernel_lib::untilize_config;
 *   untilize<width, cb_in, cb_out,
 *            InitUninitMode::Neither>(num_blocks);
 *
 * @example
 *   // Uninit only (last in sequence)
 *   using namespace compute_kernel_lib::untilize_config;
 *   untilize<width, cb_in, cb_out,
 *            InitUninitMode::UninitOnly>(num_blocks);
 */
template <
    uint32_t block_width_tiles,
    uint32_t input_cb,
    uint32_t output_cb,
    untilize_config::InitUninitMode init_uninit_mode = untilize_config::InitUninitMode::InitAndUninit,
    untilize_config::WaitMode wait_mode = untilize_config::WaitMode::WaitBlock,
    untilize_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>
ALWI void untilize(uint32_t num_blocks);

}  // namespace compute_kernel_lib

// Include implementation
#include "untilize_helpers.inl"
