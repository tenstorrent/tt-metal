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
    // Fast tilize only supports Float32 (0) and Float16_b/bfp16 (5)
    // DataFormat enum values: Float32 = 0, Float16_b = 5, Int32 = 8, etc.
    // unpack_src_format (UNPACK/MATH) and pack_dst_format (PACK) are both the L1 format
    // for the CB, equalized by JIT (genfiles.cpp:equalize_data_format_vectors).
#if defined(UCK_CHLKC_PACK)
    constexpr auto format = pack_dst_format[input_cb];
#else
    constexpr auto format = unpack_src_format[input_cb];
#endif
    return format == 0 || format == 5;  // Float32 or Float16_b (bfp16)
}

template <uint32_t input_cb, uint32_t output_cb>
constexpr bool can_use_fast_tilize() {
    return has_32x32_tiles<output_cb>() &&
           !get_dst_full_sync_enabled() &&
           has_supported_fast_tilize_format<input_cb>();
}

// =============================================================================
// CB Validation Helpers (must be called from PACK/UNPACK guards)
// =============================================================================

template <uint32_t input_cb, uint32_t output_cb>
ALWI void assert_tilize_cb_page_sizes(bool asymmetric_cb_pages) {
    const uint32_t in_page_size = get_local_cb_interface(input_cb).fifo_page_size;
    const uint32_t out_page_size = get_local_cb_interface(output_cb).fifo_page_size;
    if (asymmetric_cb_pages) {
        ASSERT(in_page_size != out_page_size);
    } else {
        ASSERT(in_page_size == out_page_size);
    }
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
    std::optional<uint32_t> total_input_pages) {

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

    // Determine if we're using fast tilize mode (automatic detection based on tile size, sync mode, and data format)
    constexpr bool use_fast = can_use_fast_tilize<input_cb, output_cb>();

    // Determine if we're doing data type reconfiguration
    constexpr bool use_unpack_reconfig =
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure) ||
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    constexpr bool use_pack_reconfig =
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::PackReconfigure) ||
        (reconfig_mode == tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure);

    const bool asymmetric_cb_pages = total_input_pages.has_value();
    ASSERT(!asymmetric_cb_pages || *total_input_pages > 0);  // total_input_pages must be > 0 when provided

    // Sanity checks: verify CB page sizes match the usage pattern.
    // Guarded because get_local_cb_interface() references cb_interface, which is
    // not defined for the MATH TRISC (trisc.cc excludes it via #if !defined(UCK_CHLKC_MATH)).
    PACK((assert_tilize_cb_page_sizes<input_cb, output_cb>(asymmetric_cb_pages)));
    PACK(ASSERT(is_valid_cb_tile_page_size(output_cb, (DataFormat)pack_dst_format[output_cb])));
    UNPACK(if (!asymmetric_cb_pages) {
        ASSERT(is_valid_cb_tile_page_size(input_cb, (DataFormat)unpack_src_format[input_cb]));
    })

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
    PACK(ASSERT(get_cb_num_pages(output_cb) >= block_width_tiles));

    // Upfront wait (when requested)
    if constexpr (wait_mode == tilize_config::WaitMode::WaitUpfront) {
        uint32_t total_wait = asymmetric_cb_pages ? *total_input_pages : (block_width_tiles * num_blocks);
        cb_wait_front(input_cb, total_wait);
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
            cb_wait_front(input_cb, input_pages);
        }

        cb_reserve_back(output_cb, block_width_tiles);

        // Assert that the contiguous read from rd_ptr won't wrap past the physical CB buffer.
        // tilize_block reads input_pages contiguous pages from rd_ptr; if rd_ptr is offset
        // (e.g. from prior use of the same CB), the read can exceed fifo_limit.
        UNPACK({
            auto& in_cb = get_local_cb_interface(input_cb);
            ASSERT(in_cb.fifo_rd_ptr + input_pages * in_cb.fifo_page_size <= in_cb.fifo_limit);
        })

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
