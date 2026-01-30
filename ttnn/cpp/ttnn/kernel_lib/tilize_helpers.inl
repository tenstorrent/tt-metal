// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Implementation file for tilize_helpers.hpp
// This file is included at the end of tilize_helpers.hpp

// Check if tile dimension arrays are available (from JIT-generated chlkc_pack_tile_dims.h)
// Tilize is a pack operation, so we use pack_tile_*_dim arrays for the output CB
#include "ttnn/kernel_lib/compute_kernel_lib_common.hpp"
#if __has_include("chlkc_pack_tile_dims.h")
#include "chlkc_pack_tile_dims.h"
#define PACK_TILE_DIMS_AVAILABLE
#endif

namespace compute_kernel_lib {

// =============================================================================
// Internal Helper Implementations
// =============================================================================

template <uint32_t cb_id>
constexpr bool has_32x32_tiles() {
#ifdef PACK_TILE_DIMS_AVAILABLE
    // Access pack tile dimensions at compile time
    constexpr uint32_t tile_r_dim = pack_tile_r_dim[cb_id];
    constexpr uint32_t tile_c_dim = pack_tile_c_dim[cb_id];

    // Fast tilize requires 32x32 tiles
    return tile_r_dim == 32 && tile_c_dim == 32;
#else
    // If header not available, assume 32x32 tiles (conservative)
    // fast_tilize already falls back to standard tilize on Blackhole
    return true;
#endif
}

template <uint32_t output_cb>
constexpr bool can_use_fast_tilize() {
    return has_32x32_tiles<output_cb>() && !get_dst_full_sync_enabled();
}

// =============================================================================
// Main Function Implementation
// =============================================================================

template <
    uint32_t input_cb,
    uint32_t output_cb,
    InitUninitMode init_uninit_mode,
    WaitMode wait_mode,
    uint32_t reconfig_from_cb>
ALWI void tilize(
    uint32_t block_width_tiles,
    uint32_t num_blocks,
    uint32_t input_pages_per_block,
    uint32_t total_input_pages) {
    // Compile-time validation
    static_assert(input_cb != output_cb, "Input and output circular buffers must be different");
    static_assert(wait_mode != WaitMode::WaitUpfront,
        "WaitUpfront is not currently supported for tilize due to lack of need. Will be added when required.");
    static_assert(!(init_uninit_mode == InitUninitMode::Neither && reconfig_from_cb != INVALID_CB),
        "reconfig_from_cb cannot be used with InitUninitMode::Neither (reconfig affects init/uninit)");

    // Derive compile-time flags from enums
    constexpr bool do_init =
        (init_uninit_mode == InitUninitMode::InitAndUninit || init_uninit_mode == InitUninitMode::InitOnly);
    constexpr bool do_uninit =
        (init_uninit_mode == InitUninitMode::InitAndUninit || init_uninit_mode == InitUninitMode::UninitOnly);
    constexpr bool use_dt_reconfig = (reconfig_from_cb != INVALID_CB);
    constexpr bool do_wait = (wait_mode == WaitMode::Wait);

    // Auto-detect if fast tilize can be used (compile-time)
    constexpr bool use_fast = can_use_fast_tilize<output_cb>();

    // =========================================================================
    // INITIALIZATION
    // =========================================================================
    if constexpr (do_init) {
        if constexpr (use_fast && use_dt_reconfig) {
            fast_tilize_init_with_dt(input_cb, block_width_tiles, output_cb);
        } else if constexpr (use_fast) {
            fast_tilize_init(input_cb, block_width_tiles, output_cb);
        } else if constexpr (use_dt_reconfig) {
            tilize_init_short_with_dt(reconfig_from_cb, input_cb, block_width_tiles, output_cb);
        } else {
            tilize_init(input_cb, block_width_tiles, output_cb);
        }
    }

    // =========================================================================
    // MAIN PROCESSING LOOP
    // =========================================================================
    if (total_input_pages > 0) {
        // Page-based mode with total pages (takes priority over input_pages_per_block)
        // Waits are chunked 32 pages at a time, last iteration may have fewer pages
        uint32_t pages_left = total_input_pages;
        constexpr uint32_t TILE_HEIGHT = 32;

        for (uint32_t i = 0; i < num_blocks; ++i) {
            uint32_t current_pages = (pages_left < TILE_HEIGHT) ? pages_left : TILE_HEIGHT;

            if constexpr (do_wait) {
                cb_wait_front(input_cb, current_pages);
            }
            cb_reserve_back(output_cb, block_width_tiles);

            if constexpr (use_fast) {
                fast_tilize_block(input_cb, block_width_tiles, output_cb);
            } else {
                tilize_block(input_cb, block_width_tiles, output_cb);
            }

            cb_push_back(output_cb, block_width_tiles);
            cb_pop_front(input_cb, current_pages);

            pages_left -= current_pages;
        }
    } else {
        // Standard or page-based mode with pages per iteration
        uint32_t input_amount = (input_pages_per_block > 0) ? input_pages_per_block : block_width_tiles;

        for (uint32_t i = 0; i < num_blocks; ++i) {
            if constexpr (do_wait) {
                cb_wait_front(input_cb, input_amount);
            }
            cb_reserve_back(output_cb, block_width_tiles);

            if constexpr (use_fast) {
                fast_tilize_block(input_cb, block_width_tiles, output_cb);
            } else {
                tilize_block(input_cb, block_width_tiles, output_cb);
            }

            cb_push_back(output_cb, block_width_tiles);
            cb_pop_front(input_cb, input_amount);
        }
    }

    // =========================================================================
    // CLEANUP
    // =========================================================================
    if constexpr (do_uninit) {
        if constexpr (use_fast) {
            fast_tilize_uninit(input_cb, output_cb);
            if constexpr (use_dt_reconfig) {
                reconfig_data_format_srca(input_cb, reconfig_from_cb);
            }
        } else if constexpr (use_dt_reconfig) {
            tilize_uninit_with_dt(input_cb, reconfig_from_cb, output_cb);
        } else {
            tilize_uninit(input_cb, output_cb);
        }
    }
}

}  // namespace compute_kernel_lib
