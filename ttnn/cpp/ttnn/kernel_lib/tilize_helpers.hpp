// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/kernel_lib_types.hpp"

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
 * - Self-documenting flag-based API
 * - Reduces code duplication across 40+ kernels
 *
 * IMPORTANT: Tilize functions require compute kernel hardware initialization.
 * You MUST call compute_kernel_hw_startup() or a functional equivalent at the
 * start of your kernel before using any tilize functions.
 *
 * Flag-Based API:
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
 *   using namespace compute_kernel_lib;
 *
 *   // Initialize compute kernel hardware FIRST
 *   compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
 *
 *   // Default behavior (most common)
 *   tilize(cb_in, 32, cb_out, 10);
 *
 *   // Fast tilize
 *   tilize<TilizeFlags::FAST>(cb_in, 32, cb_out, 10);
 *
 *   // Data type reconfiguration
 *   tilize<TilizeFlags::DT_RECONFIG>(new_cb, 16, cb_out, 5, 1, old_cb);
 *
 *   // Fast + DT combined
 *   tilize<TilizeFlags::FAST | TilizeFlags::DT_RECONFIG>(new_cb, 64, cb_out, 5, 1, old_cb);
 *
 *   // Skip wait (groupnorm pattern)
 *   tilize<TilizeFlags::SKIP_WAIT>(cb_in, per_core_N, cb_out, per_core_M);
 */

namespace compute_kernel_lib {

/**
 * @brief Unified tilize function handling ALL patterns
 *
 * This single function handles:
 * - Simple loop (subblock_h = 1)
 * - Activation pattern (subblock_h > 1)
 * - Fast variants (FAST flag)
 * - Data type reconfiguration (DT_RECONFIG flag)
 * - Variable row alignment (total_rows > 0) - for non-tile-aligned data
 * - Asymmetric input/output counts (input_count > 0) - for different wait/pop vs reserve/push
 *
 * IMPORTANT - HARDWARE INITIALIZATION REQUIREMENT:
 * Before calling this function, you MUST initialize the compute kernel hardware by
 * calling compute_kernel_hw_startup() or a functional equivalent at the start of
 * your kernel. Failure to do so will result in undefined behavior.
 *
 * @tparam flags TilizeFlags controlling behavior (default: NONE = all defaults)
 *
 * @param in_cb Input circular buffer ID (if DT_RECONFIG set, this is the NEW CB)
 * @param tiles_per_row Number of tiles per row (output reserve/push count)
 * @param out_cb Output circular buffer ID
 * @param num_blocks Number of blocks to process
 * @param subblock_height Height of each subblock in tiles (default: 1)
 * @param prev_in_cb Previous input CB for DT tracking (default: 0, only used if DT_RECONFIG set)
 * @param override_input_count Override cb_wait/pop count (default: 0 = use tiles_per_row)
 * @param total_rows Total input rows for variable alignment (default: 0 = disabled)
 *
 * @example
 *   // Default behavior (most common)
 *   tilize(cb_in, 32, cb_out, 10);
 *
 * @example
 *   // Fast tilize
 *   tilize<TilizeFlags::FAST>(cb_in, 32, cb_out, 10);
 *
 * @example
 *   // Data type reconfiguration
 *   tilize<TilizeFlags::DT_RECONFIG>(new_cb, 16, cb_out, 5, 1, old_cb);
 *
 * @example
 *   // Fast + DT combined
 *   tilize<TilizeFlags::FAST | TilizeFlags::DT_RECONFIG>(new_cb, 64, cb_out, 5, 1, old_cb);
 *
 * @example
 *   // Skip init and uninit (manual control)
 *   tilize<TilizeFlags::SKIP_INIT | TilizeFlags::SKIP_UNINIT>(cb_in, 32, cb_out, 10);
 *
 * @example
 *   // Skip wait in loop (groupnorm pattern with pre-loaded data)
 *   tilize<TilizeFlags::SKIP_WAIT>(cb_in, per_core_N, cb_out, per_core_M);
 *
 * @example
 *   // Variable row alignment (conv3d pattern)
 *   tilize(cb_in, matmul_K_t, cb_out, matmul_M_t, 1, 0, 0, num_patches);
 *
 * @example
 *   // Asymmetric input/output (convert_to_hwc pattern)
 *   tilize(cb_in, total_tiles, cb_out, 1, 1, 0, total_sticks);
 */
template <TilizeFlags flags = TilizeFlags::NONE>
ALWI void tilize(
    uint32_t in_cb,
    uint32_t tiles_per_row,
    uint32_t out_cb,
    uint32_t num_blocks,
    uint32_t subblock_height = 1,
    uint32_t prev_in_cb = 0,
    uint32_t override_input_count = 0,
    uint32_t total_rows = 0) {
    constexpr bool do_init = !has_flag(flags, TilizeFlags::SKIP_INIT);
    constexpr bool do_uninit = !has_flag(flags, TilizeFlags::SKIP_UNINIT);
    constexpr bool use_fast = has_flag(flags, TilizeFlags::FAST);
    constexpr bool use_dt = has_flag(flags, TilizeFlags::DT_RECONFIG);
    constexpr bool skip_wait = has_flag(flags, TilizeFlags::SKIP_WAIT);

    if constexpr (do_init) {
        if constexpr (use_dt && use_fast) {
            fast_tilize_init_with_dt(in_cb, tiles_per_row, out_cb);
        } else if constexpr (use_dt) {
            tilize_init_short_with_dt(prev_in_cb, in_cb, tiles_per_row, out_cb);
        } else if constexpr (use_fast) {
            fast_tilize_init(in_cb, tiles_per_row, out_cb);
        } else {
            tilize_init(in_cb, tiles_per_row, out_cb);
        }
    }

    if (total_rows > 0) {
        uint32_t rows_left = total_rows;
        constexpr uint32_t TILE_HEIGHT = 32;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            for (uint32_t h = 0; h < subblock_height; ++h) {
                uint32_t current_input = rows_left < TILE_HEIGHT ? rows_left : TILE_HEIGHT;

                if constexpr (!skip_wait) {
                    cb_wait_front(in_cb, current_input);
                }
                cb_reserve_back(out_cb, tiles_per_row);

                if constexpr (use_fast) {
                    fast_tilize_block(in_cb, tiles_per_row, out_cb);
                } else {
                    tilize_block(in_cb, tiles_per_row, out_cb);
                }

                cb_push_back(out_cb, tiles_per_row);
                cb_pop_front(in_cb, current_input);

                rows_left -= current_input;
            }
        }
    } else {
        uint32_t input_amount = (override_input_count > 0) ? override_input_count : tiles_per_row;

        for (uint32_t block = 0; block < num_blocks; ++block) {
            for (uint32_t h = 0; h < subblock_height; ++h) {
                if constexpr (!skip_wait) {
                    cb_wait_front(in_cb, input_amount);
                }
                cb_reserve_back(out_cb, tiles_per_row);

                if constexpr (use_fast) {
                    fast_tilize_block(in_cb, tiles_per_row, out_cb);
                } else {
                    tilize_block(in_cb, tiles_per_row, out_cb);
                }

                cb_push_back(out_cb, tiles_per_row);
                cb_pop_front(in_cb, input_amount);
            }
        }
    }

    if constexpr (do_uninit) {
        if constexpr (use_fast) {
            fast_tilize_uninit(in_cb, out_cb);
        } else if constexpr (use_dt) {
            tilize_uninit_with_dt(in_cb, prev_in_cb, out_cb);
        } else {
            tilize_uninit(in_cb, out_cb);
        }
    }
}

// =============================================================================
// Config-Based Tilize Functions
// =============================================================================

/**
 * @brief Initialize tilize - based on Config
 * @tparam Config TilizeConfig<InputCB<N>, OutputCB<N>, PreviousCB<N>, Flags>
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
 * @tparam Config TilizeConfig<InputCB<N>, OutputCB<N>, PreviousCB<N>, Flags>
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
 * @tparam Config TilizeConfig<InputCB<N>, OutputCB<N>, PreviousCB<N>, Flags>
 *
 * @param tiles_per_row Number of tiles per row (output reserve/push count)
 * @param num_blocks Number of blocks to process
 * @param subblock_height Height of each subblock in tiles
 * @param override_input_count Override cb_wait/pop count (0 = use tiles_per_row)
 * @param total_rows Total input rows for variable alignment (0 = disabled)
 *
 * @example
 *   // Default behavior
 *   tilize<TilizeConfig<InputCB<cb_in>, OutputCB<cb_out>>>(32, 10, 1, 0, 0);
 *
 * @example
 *   // Fast tilize
 *   tilize<TilizeConfig<InputCB<cb_in>, OutputCB<cb_out>, PreviousCB<0>,
 *                       TilizeFlags::FAST>>(32, 10, 1, 0, 0);
 *
 * @example
 *   // DT reconfiguration
 *   tilize<TilizeConfig<InputCB<new_cb>, OutputCB<cb_out>, PreviousCB<old_cb>,
 *                       TilizeFlags::DT_RECONFIG>>(16, 5, 1, 0, 0);
 */
template <typename Config>
ALWI void tilize(
    uint32_t tiles_per_row,
    uint32_t num_blocks,
    uint32_t subblock_height,
    uint32_t override_input_count,
    uint32_t total_rows) {
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
