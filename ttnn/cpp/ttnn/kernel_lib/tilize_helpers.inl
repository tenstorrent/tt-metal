// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file tilize_helpers.inl
 * @brief Implementation of tilize helper functions
 *
 * This file contains the implementation details for the tilize() function.
 * It should only be included by tilize_helpers.hpp.
 */
#include "ttnn/cpp/ttnn/kernel_lib/cb_helpers.hpp"

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
    tilize_config::InitUninitMode init_uninit_mode,
    tilize_config::WaitMode wait_mode,
    tilize_config::ReconfigureRegisterDatatypeMode reconfig_mode>
ALWI void tilize(
    uint32_t block_width_tiles,
    uint32_t num_blocks,
    uint32_t total_input_pages) {

    // Compile-time validation
    static_assert(input_cb != output_cb,
        "Tilize cannot be done in-place: input_cb and output_cb must be different");
    static_assert(input_cb < 32,
        "Invalid input_cb: must be less than 32");
    static_assert(output_cb < 32,
        "Invalid output_cb: must be less than 32");

    // Runtime parameter validation
    ASSERT(block_width_tiles > 0);
    ASSERT(num_blocks > 0);

    // Determine if we're using fast tilize mode (compile-time detection)
    constexpr bool use_fast = can_use_fast_tilize<output_cb>();

    // Determine if we're doing data type reconfiguration
    constexpr bool use_unpack_reconfig =
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure) ||
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    constexpr bool use_pack_reconfig =
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::PackReconfigure) ||
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    // Capture once — compiler propagates constexpr 0 and eliminates dead branches at symmetric call sites
    const bool asymmetric_cb_pages = (total_input_pages > 0);

    // Sanity checks: verify CB page sizes match the usage pattern
    const uint32_t in_page_size = get_local_cb_interface(input_cb).fifo_page_size;
    const uint32_t out_page_size = get_local_cb_interface(output_cb).fifo_page_size;
    if (asymmetric_cb_pages) {
        // Asymmetric: input and output CBs must have different page sizes
        ASSERT(in_page_size != out_page_size);
    } else {
        // Symmetric: page sizes must match and input must be tile-aligned
        ASSERT(in_page_size == out_page_size);
        UNPACK(ASSERT(is_valid_cb_tile_page_size(input_cb, (DataFormat)unpack_src_format[input_cb])));
    }
    // Output CB must always have tile-aligned page size
    PACK(ASSERT(is_valid_cb_tile_page_size(output_cb, (DataFormat)pack_dst_format[output_cb])));

    // Reconfigure register datatypes if requested
    if constexpr (use_unpack_reconfig) {
        // Reconfigure srcA for unpack
        reconfig_data_format_srca(input_cb);

        if constexpr (use_fast) {
            // Reconfigure srcB only in fast mode
            reconfig_data_format_srcb(input_cb);
        }
    }

    if constexpr (use_pack_reconfig) {
        // Reconfigure output for pack
        pack_reconfig_data_format(output_cb);
    }

    // Compile-time initialization based on InitUninitMode
    if constexpr (
        init_uninit_mode == tilize_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == tilize_config::InitUninitMode::InitOnly) {

        if constexpr (use_fast) {
            fast_tilize_init(input_cb, block_width_tiles, output_cb);
        } else {
            tilize_init(input_cb, block_width_tiles, output_cb);
        }
    }

    // Validate CB capacity: output CB must hold at least block_width_tiles
    ASSERT(get_cb_num_pages(output_cb) >= block_width_tiles);

    // Upfront wait (when requested)
    if constexpr (wait_mode == tilize_config::WaitMode::WaitUpfront) {
        uint32_t total_wait = asymmetric_cb_pages ? total_input_pages : (block_width_tiles * num_blocks);
        cb_wait_front(input_cb, total_wait);
    }

    // Main loop
    uint32_t pages_left = total_input_pages;  // 0 when symmetric — optimized away by compiler
    uint32_t input_pages = block_width_tiles;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Determine input pages for this block
        if (asymmetric_cb_pages) {
            // Asymmetric: min(32, pages_left)
            input_pages = (pages_left < 32) ? pages_left : 32;
        }

        if constexpr (wait_mode == tilize_config::WaitMode::WaitBlock) {
            cb_wait_front(input_cb, input_pages);
        }

        cb_reserve_back(output_cb, block_width_tiles);

        if constexpr (use_fast) {
            fast_tilize_block(input_cb, block_width_tiles, output_cb);
        } else {
            tilize_block(input_cb, block_width_tiles, output_cb);
        }

        cb_push_back(output_cb, block_width_tiles);
        cb_pop_front(input_cb, input_pages);

        if (asymmetric_cb_pages) {
            pages_left -= input_pages;
        }
    }

    // Compile-time cleanup based on InitUninitMode
    if constexpr (
        init_uninit_mode == tilize_config::InitUninitMode::InitAndUninit ||
        init_uninit_mode == tilize_config::InitUninitMode::UninitOnly) {

        if constexpr (use_fast) {
            fast_tilize_uninit(input_cb, output_cb);
        } else {
            tilize_uninit(input_cb, output_cb);
        }
    }
}

}  // namespace compute_kernel_lib
