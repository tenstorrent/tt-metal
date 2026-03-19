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
#include "experimental/circular_buffer.h"

// JIT generates chlkc_descriptors.h (not per-variable files), included via chlkc_list.h.
// The arrays are available in scope but guarded by TRISC type:
//   - unpack_src_format[] / unpack_dst_format[]   : UNPACK and MATH TRISCs (not PACK)
//   - unpack_tile_r/c_dim[]                       : UNPACK and MATH TRISCs (not PACK)
//   - pack_src_format[] / pack_dst_format[]       : PACK TRISC only
//   - pack_tile_r/c_dim[]                         : PACK TRISC only
// Note: unpack_src_format[cb] == pack_dst_format[cb] (both are L1 format, equalized by JIT).
// Note: unpack_tile_r/c_dim[cb] == pack_tile_r/c_dim[cb] (both from desc.buf_tile_r/c_dim_arr).
namespace compute_kernel_lib {

// =============================================================================
// Internal Helper Implementations
// =============================================================================

template <uint32_t cb_id>
constexpr bool has_32x32_tiles() {
    // pack_tile_r/c_dim[] available on PACK; unpack_tile_r/c_dim[] on UNPACK/MATH.
    // Both originate from the same desc.buf_tile_r/c_dim_arr, so values are identical.
#if defined(UCK_CHLKC_PACK)
    constexpr uint32_t tile_r_dim = pack_tile_r_dim[cb_id];
    constexpr uint32_t tile_c_dim = pack_tile_c_dim[cb_id];
#else
    constexpr uint32_t tile_r_dim = unpack_tile_r_dim[cb_id];
    constexpr uint32_t tile_c_dim = unpack_tile_c_dim[cb_id];
#endif
    // Fast tilize requires 32x32 tiles
    return tile_r_dim == 32 && tile_c_dim == 32;
}

template <uint32_t input_cb>
constexpr bool has_supported_fast_tilize_format() {
    // Fast tilize only supports Float32 (0) and Float16_b (5)
    // DataFormat enum values: Float32 = 0, Float16_b = 5, Int32 = 8, etc.
    // unpack_src_format (UNPACK/MATH) and pack_dst_format (PACK) are both the L1 format
    // for the CB, equalized by JIT (genfiles.cpp:equalize_data_format_vectors).
#if defined(UCK_CHLKC_PACK)
    constexpr auto format = pack_dst_format[input_cb];
#else
    constexpr auto format = unpack_src_format[input_cb];
#endif
    return format == 0 || format == 5;  // Float32 or Float16_b
}

template <uint32_t input_cb>
constexpr bool is_fp32_input_format() {
#if defined(UCK_CHLKC_PACK)
    constexpr auto format = pack_dst_format[input_cb];
#else
    constexpr auto format = unpack_src_format[input_cb];
#endif
    return format == 0;  // Float32
}

template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb>
constexpr bool can_use_fast_tilize() {
    return block_width_tiles < 256 &&
           has_32x32_tiles<output_cb>() &&
           !get_dst_full_sync_enabled() &&
           has_supported_fast_tilize_format<input_cb>();
}

// =============================================================================
// CB Validation Helpers (must be called from PACK guard)
// =============================================================================

template <uint32_t input_cb, uint32_t output_cb>
ALWI void assert_tilize_cb_page_sizes(bool asymmetric_cb_pages) {
    // When output is a block float format (e.g. Bfp8_b), the op converts from a
    // non-block input format (e.g. Float16_b), so page sizes legitimately differ.
#if defined(UCK_CHLKC_PACK)
    if constexpr (!is_block_float_format(pack_dst_format[output_cb])) {
        const uint32_t in_page_size = get_local_cb_interface(input_cb).fifo_page_size;
        const uint32_t out_page_size = get_local_cb_interface(output_cb).fifo_page_size;
        if (asymmetric_cb_pages) {
            ASSERT(in_page_size != out_page_size);
        } else {
            ASSERT(in_page_size == out_page_size);
        }
    }
#endif
}

// =============================================================================
// Main Function Implementation
// =============================================================================

template <
    uint32_t block_width_tiles,
    uint32_t input_cb,
    uint32_t output_cb,
    tilize_config::InitUninitMode init_uninit_mode,
    tilize_config::WaitMode wait_mode,
    tilize_config::ReconfigureRegisterDatatypeMode reconfig_mode,
    tilize_config::Fp32Mode fp32_mode>
ALWI void tilize(
    uint32_t num_blocks,
    std::optional<uint32_t> total_input_pages) {

    // Compile-time validation
    static_assert(block_width_tiles > 0,
        "block_width_tiles must be greater than 0");
    static_assert(input_cb != output_cb,
        "Tilize cannot be done in-place: input_cb and output_cb must be different");
    static_assert(input_cb < 32,
        "Invalid input_cb: must be less than 32");
    static_assert(output_cb < 32,
        "Invalid output_cb: must be less than 32");

    // Runtime parameter validation
    ASSERT(num_blocks > 0);

    // Determine if we're using fast tilize mode (automatic detection based on tile size, sync mode, and data format).
    // Fp32Mode::Lossless disables fast tilize only for fp32 inputs to preserve exact values
    // (fast tilize truncates fp32 → tf32). Has no effect on non-fp32 formats.
    constexpr bool lossless_fp32_override = (fp32_mode == tilize_config::Fp32Mode::Lossless) &&
                                            is_fp32_input_format<input_cb>();
    constexpr bool use_fast = can_use_fast_tilize<block_width_tiles, input_cb, output_cb>() &&
                              !lossless_fp32_override;

    // Determine if we're doing data type reconfiguration
    constexpr bool use_unpack_reconfig =
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure) ||
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    constexpr bool use_pack_reconfig =
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::PackReconfigure) ||
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    const bool asymmetric_cb_pages = total_input_pages.has_value();
    if (asymmetric_cb_pages) {
        ASSERT(*total_input_pages > (num_blocks - 1) * 32);  // at least one row in the last block
        ASSERT(*total_input_pages <= num_blocks * 32);        // rows fit within num_blocks tile-rows
    }

    // Sanity checks: verify CB page sizes match the usage pattern.
    // Guarded because get_local_cb_interface() references cb_interface, which is
    // not defined for the MATH TRISC (trisc.cc excludes it via #if !defined(UCK_CHLKC_MATH)).
    PACK((assert_tilize_cb_page_sizes<input_cb, output_cb>(asymmetric_cb_pages)));
    PACK(ASSERT(is_valid_cb_tile_page_size(output_cb, (DataFormat)pack_dst_format[output_cb])));
    UNPACK(if (!asymmetric_cb_pages) {
        ASSERT(is_valid_cb_tile_page_size(input_cb, (DataFormat)unpack_src_format[input_cb]));
    })

    // Tilize input must not be a block float format (Bfp8/4/2 and _b variants).
    // Block floats have shared exponents that break row-major-to-tile reinterpretation.
    UNPACK(ASSERT(!is_block_float_format(unpack_src_format[input_cb])));

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

    // Validate CB capacity
    if (asymmetric_cb_pages) {
        uint32_t max_in = (*total_input_pages < 32) ? *total_input_pages : 32;
        UNPACK(ASSERT(get_cb_num_pages(input_cb) >= max_in));
    } else {
        UNPACK(ASSERT(get_cb_num_pages(input_cb) >= block_width_tiles));
    }
    PACK(ASSERT(get_cb_num_pages(output_cb) >= block_width_tiles));

    // Construct experimental::CircularBuffer objects for sync operations
    experimental::CircularBuffer in_cb(input_cb);
    experimental::CircularBuffer out_cb(output_cb);

    // Upfront wait (when requested)
    if constexpr (wait_mode == tilize_config::WaitMode::WaitUpfront) {
        uint32_t total_wait = asymmetric_cb_pages ? *total_input_pages : (block_width_tiles * num_blocks);
        in_cb.wait_front(total_wait);
    }

    // Main loop
    uint32_t pages_left = total_input_pages.value_or(0);
    uint32_t input_pages = block_width_tiles;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Determine input pages for this block
        if (asymmetric_cb_pages) {
            // Asymmetric: min(32, pages_left)
            input_pages = (pages_left < 32) ? pages_left : 32;
        }

        if constexpr (wait_mode == tilize_config::WaitMode::WaitBlock) {
            in_cb.wait_front(input_pages);
        }

        out_cb.reserve_back(block_width_tiles);

        if constexpr (use_fast) {
            fast_tilize_block(input_cb, block_width_tiles, output_cb);
        } else {
            tilize_block(input_cb, block_width_tiles, output_cb);
        }

        out_cb.push_back(block_width_tiles);
        in_cb.pop_front(input_pages);

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
