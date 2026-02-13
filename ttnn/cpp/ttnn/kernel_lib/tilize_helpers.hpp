// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt-metalium/circular_buffer_constants.h"
#include "api/compute/tilize.h"
#include "api/compute/cb_api.h"
#include "internal/circular_buffer_interface.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

/**
 * @file tilize_helpers.hpp
 * @brief Header-only kernel library for tilize operations with improved type-safe API
 *
 * This library provides a unified tilize function with:
 * - Compile-time type safety via template parameters for CB IDs
 * - Descriptive enum-based configuration instead of boolean flags
 * - Automatic fast tilize detection at compile time
 * - Support for both symmetric and asymmetric CB page consumption
 *
 * Key Features:
 * - ONE function handles all tilize patterns
 * - Zero runtime overhead (all functions inlined, compile-time dispatch)
 * - Template-based compile-time optimization
 * - Self-documenting code with enums
 *
 * IMPORTANT: Tilize functions require compute kernel hardware initialization.
 * You MUST call compute_kernel_hw_startup() or equivalent before using tilize.
 *
 * Symmetric vs Asymmetric Pages:
 *
 * The terms "symmetric" and "asymmetric" refer to whether the number of pages
 * consumed from the input CB equals the number of pages produced on the output CB.
 * This is determined by whether the input and output CBs have the same page size.
 *
 *   **Symmetric** (default, total_input_pages = 0):
 *   Input and output CBs have the same page size (both tile-sized), so each
 *   block consumes and produces the same number of pages (block_width_tiles).
 *   This is the common case.
 *
 *   **Asymmetric** (total_input_pages > 0):
 *   Input and output CBs have different page sizes (e.g., input pages are
 *   row-major sticks, output pages are tiles). Because page sizes differ, the
 *   number of input pages consumed differs from the number of output tiles
 *   produced. Pass total_input_pages to specify how many input pages exist;
 *   each block then consumes min(32, pages_left) input pages while producing
 *   block_width_tiles output tiles.
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
 *
 *   // Initialize compute kernel hardware FIRST
 *   compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
 *
 *   // Symmetric — same page size on both CBs, pages consumed == pages produced
 *   compute_kernel_lib::tilize<cb_in, cb_out>(block_width_tiles, num_blocks);
 *
 *   // Asymmetric — different page sizes, pages consumed != pages produced
 *   // (e.g., input CB has row-sized pages, output CB has tile-sized pages)
 *   compute_kernel_lib::tilize<cb_in, cb_out>(out_tiles, num_blocks, total_rows);
 */

namespace compute_kernel_lib {

// Nested namespace for tilize-specific types to avoid conflicts
namespace tilize_config {

// Sentinel value for invalid/unset circular buffer ID
constexpr uint32_t INVALID_CB = NUM_CIRCULAR_BUFFERS;

/**
 * @brief Controls register datatype reconfiguration mode for tilize operations
 *
 * NoReconfigure - no register datatype reconfiguration (default)
 * UnpackReconfigure - reconfigure only unpack registers (srcA/srcB)
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
 * @brief Controls init/uninit behavior for tilize operations
 *
 * Use InitAndUninit for standalone operations (default).
 * Use InitOnly/UninitOnly/Neither when chaining multiple tilize calls.
 */
enum class InitUninitMode : uint8_t {
    InitAndUninit,  // Default - standalone operation (calls both init and uninit)
    InitOnly,       // First in a sequence (calls init only)
    UninitOnly,     // Last in a sequence (calls uninit only)
    Neither         // Middle of a sequence (calls neither)
};

/**
 * @brief Controls input synchronization (wait strategy) for tilize operations
 *
 * WaitBlock (default) - wait per block/iteration (standard behavior)
 * WaitUpfront - wait for all blocks upfront before processing
 * NoWait - caller manages synchronization (data is pre-loaded)
 */
enum class WaitMode : uint8_t {
    WaitBlock,    // Default - wait per block/iteration
    WaitUpfront,  // Wait for all blocks upfront before processing
    NoWait        // Caller manages synchronization (skip cb_wait_front)
};

}  // namespace tilize_config

/**
 * @brief Unified tilize function handling all patterns with type-safe API
 *
 * Handles symmetric and asymmetric page consumption (see file header for
 * definitions), automatic fast tilize detection, data type reconfiguration,
 * init/uninit chaining, and flexible wait strategies — all with zero runtime
 * overhead via compile-time dispatch.
 *
 * IMPORTANT - HARDWARE INITIALIZATION REQUIREMENT:
 * Before calling this function, you MUST initialize the compute kernel hardware by
 * calling compute_kernel_hw_startup() or equivalent at the start of your kernel.
 *
 * @tparam input_cb Input circular buffer ID (compile-time for type safety)
 * @tparam output_cb Output circular buffer ID (compile-time for type safety)
 * @tparam init_uninit_mode Controls init/uninit behavior (default: InitAndUninit)
 * @tparam wait_mode Controls input synchronization strategy (default: WaitBlock)
 * @tparam reconfig_mode Controls register datatype reconfiguration (default: NoReconfigure)
 *
 * @param block_width_tiles Output tiles per block
 * @param num_blocks Number of blocks to process
 * @param total_input_pages Total input pages across all blocks (0 = symmetric, >0 = asymmetric).
 *        When 0 (default): input and output CBs have the same page size, so pages consumed
 *        equals pages produced — each block consumes/produces block_width_tiles CB pages.
 *        When >0: input and output CBs have different page sizes, so pages consumed differs
 *        from pages produced — each block up to 32 input pages while
 *        producing block_width_tiles output tiles.
 *
 * @example
 *   // Symmetric — same page size, pages consumed == pages produced.
 *   tilize<cb_in, cb_out>(32, num_blocks);
 *
 * @example
 *   // Asymmetric — different page sizes, pages consumed != pages produced.
 *   // Input CB has row-sized pages, output CB has tile-sized pages.
 *   // Each block consumes up to 32 input pages and produces out_tiles_per_block output tiles.
 *   tilize<cb_in, cb_out>(out_tiles_per_block, num_blocks, total_rows);
 *
 * @example
 *   // Asymmetric, single block — all input rows tilized at once.
 *   tilize<cb_in, cb_out>(total_out_tiles, 1, total_rows);
 *
 * @example
 *   // Data type reconfiguration (unpack + pack registers)
 *   using namespace compute_kernel_lib::tilize_config;
 *   tilize<new_cb, cb_out,
 *          InitUninitMode::InitAndUninit,
 *          WaitMode::WaitBlock,
 *          ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(16, 5);
 *
 * @example
 *   // Caller manages synchronization (data already in CB)
 *   using namespace compute_kernel_lib::tilize_config;
 *   tilize<cb_in, cb_out,
 *          InitUninitMode::InitAndUninit,
 *          WaitMode::NoWait>(block_w, num_blocks);
 *
 * @example
 *   // Chained tilize calls — first/middle/last init control
 *   using namespace compute_kernel_lib::tilize_config;
 *   tilize<cb_in, cb_out, InitUninitMode::InitOnly>(w, blocks);   // first
 *   tilize<cb_in, cb_out, InitUninitMode::Neither>(w, blocks);    // middle
 *   tilize<cb_in, cb_out, InitUninitMode::UninitOnly>(w, blocks); // last
 */
template <
    uint32_t input_cb,
    uint32_t output_cb,
    tilize_config::InitUninitMode init_uninit_mode = tilize_config::InitUninitMode::InitAndUninit,
    tilize_config::WaitMode wait_mode = tilize_config::WaitMode::WaitBlock,
    tilize_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>
ALWI void tilize(uint32_t block_width_tiles, uint32_t num_blocks, uint32_t total_input_pages = 0);

}  // namespace compute_kernel_lib

// Include implementation
#include "tilize_helpers.inl"
