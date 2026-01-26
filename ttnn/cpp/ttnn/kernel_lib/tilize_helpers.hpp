// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/tilize.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/kernel_lib_types.hpp"

/**
 * @file tilize_helpers.hpp
 * @brief Header-only kernel library for tilize operations
 *
 * This library provides a unified config-based function for ALL tilize operations.
 *
 * Key Features:
 * - Config-based API with compile-time CB indices
 * - Zero runtime overhead (all functions inlined)
 * - Template-based compile-time optimization
 * - Self-documenting named template parameters
 * - Reduces code duplication across 40+ kernels
 *
 * IMPORTANT: Tilize functions require compute kernel hardware initialization.
 * You MUST call compute_kernel_hw_startup() or a functional equivalent at the
 * start of your kernel before using any tilize functions.
 *
 * Flag-Based Configuration:
 *   Flags represent DEVIATIONS from default behavior.
 *   TilizeFlags::NONE = all defaults (init, uninit, no fast, no dt, do wait)
 *
 *   | Flag        | Meaning                          | Default without flag |
 *   |-------------|----------------------------------|----------------------|
 *   | SKIP_INIT   | Don't call tilize_init           | Do init              |
 *   | SKIP_UNINIT | Don't call tilize_uninit         | Do uninit            |
 *   | FAST        | Use fast_tilize_* functions      | Standard tilize      |
 *   | DT_RECONFIG | Enable data type reconfiguration | Disabled             |
 *   | SKIP_WAIT   | Skip cb_wait_front in loop       | Do wait              |
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
 *
 *   // Initialize compute kernel hardware FIRST
 *   compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
 *
 *   // Default behavior (most common) - subblock_height, override_input_count, total_rows default to 1, 0, 0
 *   tilize<TilizeConfig<InputCB<cb_in>, OutputCB<cb_out>>>(tiles_per_row, num_blocks);
 *
 *   // Fast tilize (no need to specify PreviousCB)
 *   tilize<TilizeConfig<InputCB<cb_in>, OutputCB<cb_out>, TilizeFlags::FAST>>(tiles_per_row, num_blocks);
 *
 *   // Skip wait (groupnorm pattern)
 *   tilize<TilizeConfig<InputCB<cb_in>, OutputCB<cb_out>, TilizeFlags::SKIP_WAIT>>(per_core_N, per_core_M);
 *
 *   // With subblock_height (activation pattern)
 *   tilize<TilizeConfig<InputCB<cb_in>, OutputCB<cb_out>>>(Wt, 1, Ht);
 *
 *   // Data type reconfiguration (requires PreviousCB)
 *   tilize<TilizeConfig<InputCB<new_cb>, OutputCB<cb_out>,
 *                       TilizeFlags::DT_RECONFIG, PreviousCB<old_cb>>>(tiles_per_row, num_blocks);
 */

namespace compute_kernel_lib {

// =============================================================================
// Config-Based Tilize Functions
// =============================================================================

/**
 * @brief Initialize tilize - based on Config
 * @tparam Config TilizeConfig<InputCB<N>, OutputCB<N>, Flags, PreviousCB<N>>
 */
template <typename Config>
ALWI void tilize_init(uint32_t tiles_per_row) {
    static_assert(
        std::is_base_of_v<TilizeConfigBase, Config>,
        "Config must derive from TilizeConfigBase (use TilizeConfig<InputCB<N>, OutputCB<N>>)");

    constexpr uint32_t input_cb = Config::input_cb;
    constexpr uint32_t output_cb = Config::output_cb;
    constexpr uint32_t previous_cb = Config::previous_cb;
    constexpr TilizeFlags flags = Config::flags;

    constexpr bool use_fast = has_flag(flags, TilizeFlags::FAST);
    constexpr bool use_dt = has_flag(flags, TilizeFlags::DT_RECONFIG);

    if constexpr (use_dt && use_fast) {
        fast_tilize_init_with_dt(input_cb, tiles_per_row, output_cb);
    } else if constexpr (use_dt) {
        tilize_init_short_with_dt(previous_cb, input_cb, tiles_per_row, output_cb);
    } else if constexpr (use_fast) {
        fast_tilize_init(input_cb, tiles_per_row, output_cb);
    } else {
        ::tilize_init(input_cb, tiles_per_row, output_cb);
    }
}

/**
 * @brief Uninitialize tilize - based on Config
 * @tparam Config TilizeConfig<InputCB<N>, OutputCB<N>, Flags, PreviousCB<N>>
 */
template <typename Config>
ALWI void tilize_uninit() {
    static_assert(
        std::is_base_of_v<TilizeConfigBase, Config>,
        "Config must derive from TilizeConfigBase (use TilizeConfig<InputCB<N>, OutputCB<N>>)");

    constexpr uint32_t input_cb = Config::input_cb;
    constexpr uint32_t output_cb = Config::output_cb;
    constexpr uint32_t previous_cb = Config::previous_cb;
    constexpr TilizeFlags flags = Config::flags;

    constexpr bool use_fast = has_flag(flags, TilizeFlags::FAST);
    constexpr bool use_dt = has_flag(flags, TilizeFlags::DT_RECONFIG);

    if constexpr (use_fast) {
        fast_tilize_uninit(input_cb, output_cb);
    } else if constexpr (use_dt) {
        tilize_uninit_with_dt(input_cb, previous_cb, output_cb);
    } else {
        ::tilize_uninit(input_cb, output_cb);
    }
}

/**
 * @brief Config-based tilize function
 *
 * @tparam Config TilizeConfig<InputCB<N>, OutputCB<N>, Flags, PreviousCB<N>>
 *
 * @param tiles_per_row Number of tiles per row (output reserve/push count)
 * @param num_blocks Number of blocks to process
 * @param subblock_height Height of each subblock in tiles
 * @param override_input_count Override cb_wait/pop count (0 = use tiles_per_row)
 * @param total_rows Total input rows for variable alignment (0 = disabled)
 *
 * @example
 *   // Default behavior
 *   tilize<TilizeConfig<InputCB<cb_in>, OutputCB<cb_out>>>(32, 10);
 *
 * @example
 *   // Fast tilize (no need to specify PreviousCB)
 *   tilize<TilizeConfig<InputCB<cb_in>, OutputCB<cb_out>, TilizeFlags::FAST>>(32, 10);
 *
 * @example
 *   // With subblock_height (activation pattern)
 *   tilize<TilizeConfig<InputCB<cb_in>, OutputCB<cb_out>>>(Wt, 1, Ht);
 *
 * @example
 *   // DT reconfiguration (requires PreviousCB)
 *   tilize<TilizeConfig<InputCB<new_cb>, OutputCB<cb_out>,
 *                       TilizeFlags::DT_RECONFIG, PreviousCB<old_cb>>>(16, 5);
 */
template <typename Config>
ALWI void tilize(
    uint32_t tiles_per_row,
    uint32_t num_blocks,
    uint32_t subblock_height = 1,
    uint32_t override_input_count = 0,
    uint32_t total_rows = 0) {
    static_assert(
        std::is_base_of_v<TilizeConfigBase, Config>,
        "Config must derive from TilizeConfigBase (use TilizeConfig<InputCB<N>, OutputCB<N>>)");

    constexpr uint32_t input_cb = Config::input_cb;
    constexpr uint32_t output_cb = Config::output_cb;
    constexpr uint32_t previous_cb = Config::previous_cb;
    constexpr TilizeFlags flags = Config::flags;

    constexpr bool do_init = !has_flag(flags, TilizeFlags::SKIP_INIT);
    constexpr bool do_uninit = !has_flag(flags, TilizeFlags::SKIP_UNINIT);
    constexpr bool use_fast = has_flag(flags, TilizeFlags::FAST);
    constexpr bool use_dt = has_flag(flags, TilizeFlags::DT_RECONFIG);
    constexpr bool skip_wait = has_flag(flags, TilizeFlags::SKIP_WAIT);

    if constexpr (do_init) {
        if constexpr (use_dt && use_fast) {
            fast_tilize_init_with_dt(input_cb, tiles_per_row, output_cb);
        } else if constexpr (use_dt) {
            tilize_init_short_with_dt(previous_cb, input_cb, tiles_per_row, output_cb);
        } else if constexpr (use_fast) {
            fast_tilize_init(input_cb, tiles_per_row, output_cb);
        } else {
            ::tilize_init(input_cb, tiles_per_row, output_cb);
        }
    }

    if (total_rows > 0) {
        uint32_t rows_left = total_rows;
        constexpr uint32_t TILE_HEIGHT = 32;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            for (uint32_t h = 0; h < subblock_height; ++h) {
                uint32_t current_input = rows_left < TILE_HEIGHT ? rows_left : TILE_HEIGHT;

                if constexpr (!skip_wait) {
                    cb_wait_front(input_cb, current_input);
                }
                cb_reserve_back(output_cb, tiles_per_row);

                if constexpr (use_fast) {
                    fast_tilize_block(input_cb, tiles_per_row, output_cb);
                } else {
                    tilize_block(input_cb, tiles_per_row, output_cb);
                }

                cb_push_back(output_cb, tiles_per_row);
                cb_pop_front(input_cb, current_input);

                rows_left -= current_input;
            }
        }
    } else {
        uint32_t input_amount = (override_input_count > 0) ? override_input_count : tiles_per_row;

        for (uint32_t block = 0; block < num_blocks; ++block) {
            for (uint32_t h = 0; h < subblock_height; ++h) {
                if constexpr (!skip_wait) {
                    cb_wait_front(input_cb, input_amount);
                }
                cb_reserve_back(output_cb, tiles_per_row);

                if constexpr (use_fast) {
                    fast_tilize_block(input_cb, tiles_per_row, output_cb);
                } else {
                    tilize_block(input_cb, tiles_per_row, output_cb);
                }

                cb_push_back(output_cb, tiles_per_row);
                cb_pop_front(input_cb, input_amount);
            }
        }
    }

    if constexpr (do_uninit) {
        if constexpr (use_fast) {
            fast_tilize_uninit(input_cb, output_cb);
        } else if constexpr (use_dt) {
            tilize_uninit_with_dt(input_cb, previous_cb, output_cb);
        } else {
            ::tilize_uninit(input_cb, output_cb);
        }
    }
}

}  // namespace compute_kernel_lib

// Make config types available without namespace prefix when header is included
using compute_kernel_lib::InputCB;
using compute_kernel_lib::OutputCB;
using compute_kernel_lib::PreviousCB;
using compute_kernel_lib::TilizeConfig;
using compute_kernel_lib::TilizeFlags;
