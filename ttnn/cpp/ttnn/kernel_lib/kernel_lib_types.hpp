// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

/**
 * @file kernel_lib_types.hpp
 * @brief Type-safe template parameter helpers for tilize/untilize operations
 *
 * This file provides:
 * 1. Wrapper types for self-documenting numeric template parameters
 * 2. Bit flags for boolean options (deviation-based: flags represent non-default values)
 *
 * Design Principle:
 *   Flags represent DEVIATIONS from default behavior.
 *   - If a parameter's default is `true`, the flag represents the `false` case.
 *   - If a parameter's default is `false`, the flag represents the `true` case.
 *   - `Flags::NONE` = all defaults = most common usage.
 *
 * Usage:
 *   // BEFORE (hard to parse):
 *   tilize<true, true, true, true, false>(cb_in, 32, cb_out, 10);
 *   untilize<32, cb_in, cb_out, true, true, true>(num_rows);
 *
 *   // AFTER (self-documenting):
 *   tilize<TilizeFlags::FAST | TilizeFlags::DT_RECONFIG>(cb_in, 32, cb_out, 10);
 *   untilize<UntilizeConfig<WidthInTiles<32>, InputCB<cb_in>, OutputCB<cb_out>,
 *                           UntilizeFlags::WAIT_UPFRONT>>(num_rows);
 */

namespace compute_kernel_lib {

// =============================================================================
// Wrapper Types for Self-Documenting Template Parameters
// =============================================================================

/**
 * @brief Wrapper for width in tiles (number of tiles per row)
 * @tparam N The width value in tiles
 */
template <uint32_t N>
struct WidthInTiles {
    static constexpr uint32_t value = N;
};

/**
 * @brief Wrapper for input circular buffer index
 * @tparam N The circular buffer index (0-31)
 */
template <uint32_t N>
struct InputCB {
    static constexpr uint32_t value = N;
};

/**
 * @brief Wrapper for output circular buffer index
 * @tparam N The circular buffer index (0-31)
 */
template <uint32_t N>
struct OutputCB {
    static constexpr uint32_t value = N;
};

/**
 * @brief Wrapper for previous circular buffer index (used for DT reconfiguration)
 * @tparam N The circular buffer index (0-31)
 */
template <uint32_t N>
struct PreviousCB {
    static constexpr uint32_t value = N;
};

// Type-safe value extractors - only defined for valid wrapper types
// Using wrong type causes "incomplete type" compile error
template <typename T>
struct ExtractWidthInTiles;
template <uint32_t N>
struct ExtractWidthInTiles<WidthInTiles<N>> {
    static constexpr uint32_t value = N;
};

template <typename T>
struct ExtractInputCB;
template <uint32_t N>
struct ExtractInputCB<InputCB<N>> {
    static constexpr uint32_t value = N;
};

template <typename T>
struct ExtractOutputCB;
template <uint32_t N>
struct ExtractOutputCB<OutputCB<N>> {
    static constexpr uint32_t value = N;
};

template <typename T>
struct ExtractPreviousCB;
template <uint32_t N>
struct ExtractPreviousCB<PreviousCB<N>> {
    static constexpr uint32_t value = N;
};

// =============================================================================
// Tilize Flags - Represent DEVIATIONS from default behavior
// =============================================================================
//
// Defaults: init=true, uninit=true, fast=false, dt=false, skip_wait=false
// Flags are only set when NON-DEFAULT value is needed.
//
// | Flag        | Meaning                          | Old param          |
// |-------------|----------------------------------|---------------------|
// | NONE        | All defaults (most common)       | <true,true,false,false,false> |
// | SKIP_INIT   | Don't call tilize_init           | init=false          |
// | SKIP_UNINIT | Don't call tilize_uninit         | uninit=false        |
// | FAST        | Use fast_tilize_* functions      | use_fast=true       |
// | DT_RECONFIG | Enable data type reconfiguration | use_dt=true         |
// | SKIP_WAIT   | Skip cb_wait_front in loop       | skip_wait=true      |

enum class TilizeFlags : uint32_t {
    NONE = 0,              // All defaults (most common)
    SKIP_INIT = 1 << 0,    // Don't call tilize_init (default: do init)
    SKIP_UNINIT = 1 << 1,  // Don't call tilize_uninit (default: do uninit)
    FAST = 1 << 2,         // Use fast_tilize_* (default: standard)
    DT_RECONFIG = 1 << 3,  // Enable DT reconfiguration (default: disabled)
    SKIP_WAIT = 1 << 4,    // Skip cb_wait_front in loop (default: do wait)
};

constexpr TilizeFlags operator|(TilizeFlags a, TilizeFlags b) {
    return static_cast<TilizeFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

constexpr TilizeFlags operator&(TilizeFlags a, TilizeFlags b) {
    return static_cast<TilizeFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

constexpr bool has_flag(TilizeFlags flags, TilizeFlags flag) {
    return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(flag)) != 0;
}

// =============================================================================
// Untilize Flags
// =============================================================================
//
// | Flag                          | Meaning                                         |
// |-------------------------------|-------------------------------------------------|
// | NONE                          | All defaults                                    |
// | SKIP_INIT                     | Don't call init (default: do init)              |
// | SKIP_UNINIT                   | Don't call uninit (default: do uninit)          |
// | WAIT_UPFRONT                  | Wait for all tiles upfront (default: per-row)   |
// | FORCE_PACK_UNTILIZE_WIDE_FP32 | Force block-based pack_untilize for wide FP32   |

enum class UntilizeFlags : uint32_t {
    NONE = 0,                                // All defaults
    SKIP_INIT = 1 << 0,                      // Don't call init (default: do init)
    SKIP_UNINIT = 1 << 1,                    // Don't call uninit (default: do uninit)
    WAIT_UPFRONT = 1 << 2,                   // Wait for all tiles upfront (default: per-row)
    FORCE_PACK_UNTILIZE_WIDE_FP32 = 1 << 3,  // Force block-based pack_untilize for wide FP32
};

constexpr UntilizeFlags operator|(UntilizeFlags a, UntilizeFlags b) {
    return static_cast<UntilizeFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

constexpr UntilizeFlags operator&(UntilizeFlags a, UntilizeFlags b) {
    return static_cast<UntilizeFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

constexpr bool has_flag(UntilizeFlags flags, UntilizeFlags flag) {
    return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(flag)) != 0;
}

// =============================================================================
// Config Base Classes - Define the required interface/contract
// =============================================================================
//
// Base classes document what static constexpr members derived configs must provide.
// Functions use std::is_base_of to validate config types at compile time.

/**
 * Base class for untilize configuration.
 * Derived classes MUST provide these static constexpr members:
 *   - width_in_tiles : uint32_t  (number of tiles per row)
 *   - input_cb       : uint32_t  (input circular buffer index, 0-31)
 *   - output_cb      : uint32_t  (output circular buffer index, 0-31)
 *   - flags          : UntilizeFlags
 */
struct UntilizeConfigBase {
    // Marker base - enables std::is_base_of checks
};

/**
 * Base class for tilize configuration.
 * Derived classes MUST provide these static constexpr members:
 *   - input_cb    : uint32_t  (input circular buffer index, 0-31)
 *   - output_cb   : uint32_t  (output circular buffer index, 0-31)
 *   - previous_cb : uint32_t  (previous CB for DT reconfig, 0-31)
 *   - flags       : TilizeFlags
 */
struct TilizeConfigBase {
    // Marker base - enables std::is_base_of checks
};

// =============================================================================
// Config Structs - Templated children with actual values
// =============================================================================

/**
 * Configuration for untilize operations.
 * Inherits from UntilizeConfigBase; wrapper types provide named parameters.
 *
 * Usage:
 *   untilize<UntilizeConfig<WidthInTiles<4>, InputCB<cb_in>, OutputCB<cb_out>>>(num_rows);
 *
 *   // With flags:
 *   untilize<UntilizeConfig<WidthInTiles<4>, InputCB<cb_in>, OutputCB<cb_out>,
 *                           UntilizeFlags::WAIT_UPFRONT>>(num_rows, 1, total);
 *
 *   // Force block-based pack_untilize for wide FP32 (recommended for correctness):
 *   untilize<UntilizeConfig<WidthInTiles<8>, InputCB<cb_in>, OutputCB<cb_out>,
 *                           UntilizeFlags::FORCE_PACK_UNTILIZE_WIDE_FP32>>(num_rows);
 */
template <typename WidthInTilesT, typename InputCBT, typename OutputCBT, UntilizeFlags Flags = UntilizeFlags::NONE>
struct UntilizeConfig : UntilizeConfigBase {
    static constexpr uint32_t width_in_tiles = ExtractWidthInTiles<WidthInTilesT>::value;
    static constexpr uint32_t input_cb = ExtractInputCB<InputCBT>::value;
    static constexpr uint32_t output_cb = ExtractOutputCB<OutputCBT>::value;
    static constexpr UntilizeFlags flags = Flags;

    // Static validations
    static_assert(width_in_tiles > 0, "width_in_tiles must be greater than 0");
    static_assert(input_cb <= 31, "input_cb must be in range 0-31");
    static_assert(output_cb <= 31, "output_cb must be in range 0-31");
    static_assert(input_cb != output_cb, "input_cb and output_cb must be different circular buffers");
};

/**
 * Configuration for tilize operations.
 * Inherits from TilizeConfigBase; wrapper types provide named parameters.
 *
 * Template parameter order optimized for common usage (Flags before PreviousCB):
 *   - Most common: just InputCB + OutputCB
 *   - Second common: with Flags (e.g., SKIP_WAIT, FAST)
 *   - Least common: with PreviousCB (only needed for DT_RECONFIG)
 *
 * Usage:
 *   // Default (most common)
 *   tilize<TilizeConfig<InputCB<cb_in>, OutputCB<cb_out>>>(block_w, num_blocks);
 *
 *   // With flags (no need to specify PreviousCB)
 *   tilize<TilizeConfig<InputCB<cb_in>, OutputCB<cb_out>, TilizeFlags::SKIP_WAIT>>(block_w, num_blocks);
 *
 *   // With DT reconfig (requires PreviousCB)
 *   tilize<TilizeConfig<InputCB<new_cb>, OutputCB<cb_out>,
 *                       TilizeFlags::DT_RECONFIG, PreviousCB<old_cb>>>(block_w, num_blocks);
 */
template <
    typename InputCBT,
    typename OutputCBT,
    TilizeFlags Flags = TilizeFlags::NONE,
    typename PreviousCBT = PreviousCB<0>>
struct TilizeConfig : TilizeConfigBase {
    static constexpr uint32_t input_cb = ExtractInputCB<InputCBT>::value;
    static constexpr uint32_t output_cb = ExtractOutputCB<OutputCBT>::value;
    static constexpr TilizeFlags flags = Flags;
    static constexpr uint32_t previous_cb = ExtractPreviousCB<PreviousCBT>::value;

    // Static validations
    static_assert(input_cb <= 31, "input_cb must be in range 0-31");
    static_assert(output_cb <= 31, "output_cb must be in range 0-31");
    static_assert(previous_cb <= 31, "previous_cb must be in range 0-31");
    static_assert(input_cb != output_cb, "input_cb and output_cb must be different circular buffers");
};

// =============================================================================
// Backward Compatibility Aliases
// =============================================================================
// Short aliases for the wrapper types - allows existing code to compile
// while transitioning to the more descriptive names.

template <uint32_t N>
using TileWidth = WidthInTiles<N>;

template <uint32_t N>
using InCB = InputCB<N>;

template <uint32_t N>
using OutCB = OutputCB<N>;

template <uint32_t N>
using OldCB = PreviousCB<N>;

// Extractors also work with aliases (via the primary types)

}  // namespace compute_kernel_lib
