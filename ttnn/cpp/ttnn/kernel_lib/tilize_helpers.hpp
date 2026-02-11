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
 * - Clean API for non-tile-aligned circular buffer configurations
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
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
 *
 *   // Initialize compute kernel hardware FIRST
 *   compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
 *
 *   // Simple standard tilize
 *   compute_kernel_lib::tilize<cb_in, cb_out>(32, num_blocks);
 *
 *   // With unpack and pack data type reconfiguration
 *   compute_kernel_lib::tilize<new_cb, cb_out,
 *       InitUninitMode::InitAndUninit,
 *       WaitMode::WaitBlock,
 *       ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(16, num_blocks);
 *
 *   // Non-tile-aligned: per-iteration pages
 *   compute_kernel_lib::tilize<cb_in, cb_out>(
 *       total_tiles,
 *       1,
 *       NonTileAlignedCBConfig::per_iteration(total_sticks));
 *
 *   // Non-tile-aligned: total batched pages
 *   compute_kernel_lib::tilize<cb_in, cb_out>(
 *       matmul_K_t,
 *       matmul_M_t,
 *       NonTileAlignedCBConfig::total_batched(num_patches));
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

/**
 * @brief Controls non-tile-aligned circular buffer configuration for tilize
 *
 * Disabled (default) - standard tile-aligned operation
 * PerIteration - custom pages per iteration (for asymmetric input/output counts)
 * TotalBatched - total pages batched in chunks of 32 (for variable row alignment)
 */
enum class NonTileAlignedMode : uint8_t {
    Disabled,      // Standard tile-aligned (default)
    PerIteration,  // Custom pages per iteration (asymmetric pattern)
    TotalBatched   // Total pages batched in chunks of 32 (variable row pattern)
};

/**
 * @brief Configuration for non-tile-aligned circular buffer wait operations
 *
 * Use this configuration when the input circular buffer (the CB on which cb_wait_front
 * is called) does not have tile-aligned page size. This happens when the input data
 * has a page size that doesn't match the standard tile dimensions, requiring special
 * handling of wait operations.
 *
 * The struct provides three modes:
 *
 * 1. **Disabled** (default): Standard tile-aligned operation
 *    - Input CB has tile-aligned pages
 *    - Waits for block_width_tiles per iteration
 *    - Use: NonTileAlignedCBWaitConfig::disabled()
 *
 * 2. **PerIteration**: Custom pages per iteration (asymmetric input/output)
 *    - Input CB has non-tile-aligned page size
 *    - Waits for a specific number of pages per iteration (different from block_width_tiles)
 *    - Example: convert_to_hwc where input is row-major sticks, output is tiles
 *    - Use: NonTileAlignedCBWaitConfig::per_iteration(num_pages)
 *
 * 3. **TotalBatched**: Total pages batched in chunks (variable row alignment)
 *    - Input CB has non-tile-aligned page size
 *    - Waits for 32 rows (TILE_HEIGHT) per iteration until total_pages is reached
 *    - Last iteration may have fewer than 32 rows if total_pages % 32 != 0
 *    - Example: conv3d where input rows may not align to tile boundaries
 *    - Use: NonTileAlignedCBWaitConfig::total_batched(total_pages)
 *
 * Replaces the old input_pages_per_block + total_input_pages parameters
 * with a cleaner enum-based API.
 *
 * Usage:
 *   // Disabled (default - tile-aligned input CB)
 *   auto config = NonTileAlignedCBWaitConfig::disabled();
 *
 *   // Per-iteration pages (asymmetric input/output - e.g., convert_to_hwc)
 *   auto config = NonTileAlignedCBWaitConfig::per_iteration(total_sticks);
 *
 *   // Total batched pages (variable row alignment - e.g., conv3d)
 *   auto config = NonTileAlignedCBWaitConfig::total_batched(num_patches);
 */
struct NonTileAlignedCBWaitConfig {
    NonTileAlignedMode mode;
    uint32_t value;  // pages_per_iteration or total_pages depending on mode

private:
    // Private constructor - use static factory methods instead
    constexpr NonTileAlignedCBWaitConfig(NonTileAlignedMode m, uint32_t v) : mode(m), value(v) {}

public:
    // Named constructors for clarity
    static constexpr NonTileAlignedCBWaitConfig disabled() {
        return NonTileAlignedCBWaitConfig{NonTileAlignedMode::Disabled, 0};
    }

    static constexpr NonTileAlignedCBWaitConfig per_iteration(uint32_t pages) {
        return NonTileAlignedCBWaitConfig{NonTileAlignedMode::PerIteration, pages};
    }

    static constexpr NonTileAlignedCBWaitConfig total_batched(uint32_t total_pages) {
        return NonTileAlignedCBWaitConfig{NonTileAlignedMode::TotalBatched, total_pages};
    }
};

}  // namespace tilize_config

/**
 * @brief Unified tilize function handling ALL patterns with type-safe API
 *
 * This single function handles:
 * - Automatic fast tilize mode (compile-time detection)
 * - Data type reconfiguration (via reconfig_from_cb template parameter)
 * - Variable row alignment (via NonTileAlignedCBWaitConfig::total_batched)
 * - Asymmetric input/output counts (via NonTileAlignedCBWaitConfig::per_iteration)
 * - Init/uninit chaining (via InitUninitMode enum)
 * - Flexible wait strategies (via WaitMode enum)
 *
 * IMPORTANT - HARDWARE INITIALIZATION REQUIREMENT:
 * Before calling this function, you MUST initialize the compute kernel hardware by
 * calling compute_kernel_hw_startup() or equivalent at the start of your kernel.
 * Failure to do so will result in undefined behavior.
 *
 * Example:
 *   compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
 *   compute_kernel_lib::tilize<cb_in, cb_out>(block_w, num_blocks);
 *
 * @tparam input_cb Input circular buffer ID (compile-time for type safety)
 * @tparam output_cb Output circular buffer ID (compile-time for type safety)
 * @tparam init_uninit_mode Controls init/uninit behavior (default: InitAndUninit)
 * @tparam wait_mode Controls input synchronization strategy (default: Wait)
 * @tparam reconfig_mode Controls register datatype reconfiguration (default: NoReconfigure)
 *
 * @param block_width_tiles Block width in tiles (FIRST runtime argument for consistency)
 * @param num_blocks Number of blocks to process
 * @param config Non-tile-aligned CB wait configuration (default: disabled)
 *
 * @example
 *   // Simple standard tilize
 *   tilize<cb_in, cb_out>(32, 10);
 *
 * @example
 *   // Unpack and pack data type reconfiguration
 *   using namespace compute_kernel_lib::tilize_config;
 *   tilize<new_cb, cb_out,
 *          InitUninitMode::InitAndUninit,
 *          WaitMode::WaitBlock,
 *          ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(16, 5);
 *
 * @example
 *   // Only unpack reconfiguration (for srcA/srcB registers)
 *   using namespace compute_kernel_lib::tilize_config;
 *   tilize<new_cb, cb_out,
 *          InitUninitMode::InitAndUninit,
 *          WaitMode::WaitBlock,
 *          ReconfigureRegisterDatatypeMode::UnpackReconfigure>(16, 5);
 *
 * @example
 *   // Only pack reconfiguration (for output register)
 *   using namespace compute_kernel_lib::tilize_config;
 *   tilize<new_cb, cb_out,
 *          InitUninitMode::InitAndUninit,
 *          WaitMode::WaitBlock,
 *          ReconfigureRegisterDatatypeMode::PackReconfigure>(16, 5);
 *
 * @example
 *   // Per-iteration pages (asymmetric input/output - convert_to_hwc pattern)
 *   tilize<cb_in, cb_out>(
 *       total_tiles,
 *       1,
 *       tilize_config::NonTileAlignedCBWaitConfig::per_iteration(total_sticks));
 *
 * @example
 *   // Total batched pages (variable row alignment - conv3d pattern)
 *   tilize<cb_in, cb_out>(
 *       matmul_K_t,
 *       matmul_M_t,
 *       tilize_config::NonTileAlignedCBWaitConfig::total_batched(num_patches));
 *
 * @example
 *   // Skip wait (groupnorm pattern with pre-loaded data)
 *   using namespace compute_kernel_lib::tilize_config;
 *   tilize<cb_in, cb_out,
 *          InitUninitMode::InitAndUninit,
 *          WaitMode::NoWait>(per_core_N, per_core_M);
 *
 * @example
 *   // Init only (first in sequence)
 *   using namespace compute_kernel_lib::tilize_config;
 *   tilize<cb_in, cb_out,
 *          InitUninitMode::InitOnly>(width, blocks);
 *
 * @example
 *   // Neither init nor uninit (middle of sequence)
 *   using namespace compute_kernel_lib::tilize_config;
 *   tilize<cb_in, cb_out,
 *          InitUninitMode::Neither>(width, blocks);
 *
 * @example
 *   // Uninit only (last in sequence)
 *   using namespace compute_kernel_lib::tilize_config;
 *   tilize<cb_in, cb_out,
 *          InitUninitMode::UninitOnly>(width, blocks);
 */
template <
    uint32_t input_cb,
    uint32_t output_cb,
    tilize_config::InitUninitMode init_uninit_mode = tilize_config::InitUninitMode::InitAndUninit,
    tilize_config::WaitMode wait_mode = tilize_config::WaitMode::WaitBlock,
    tilize_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>
ALWI void tilize(
    uint32_t block_width_tiles,
    uint32_t num_blocks,
    tilize_config::NonTileAlignedCBWaitConfig config = tilize_config::NonTileAlignedCBWaitConfig::disabled());

}  // namespace compute_kernel_lib

// Include implementation
#include "tilize_helpers.inl"
