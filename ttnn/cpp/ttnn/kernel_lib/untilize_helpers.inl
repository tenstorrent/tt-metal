// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Implementation file for untilize_helpers.hpp
// This file is included at the end of untilize_helpers.hpp

// Check if unpack_dst_format array is available (from JIT-generated chlkc_unpack_data_format.h)
// This header is included via chlkc_list.h in the firmware build
#if __has_include("chlkc_unpack_data_format.h")
#include "chlkc_unpack_data_format.h"
#define UNPACK_DATA_FORMAT_AVAILABLE
#endif

namespace compute_kernel_lib {

// =============================================================================
// Data Format Detection Implementations
// =============================================================================

template <uint32_t cb_id>
constexpr bool is_integer_format() {
#ifdef UNPACK_DATA_FORMAT_AVAILABLE
    // Access the format at compile time
    constexpr uint32_t format = unpack_dst_format[cb_id];

    // Check if format is one of the integer types
    // Integer formats from tt_metal/hw/inc/tt-1xx/blackhole/tensix_types.h:
    return format == 8 ||   // Int32
           format == 24 ||  // UInt32
           format == 14 ||  // Int8
           format == 30 ||  // UInt8
           format == 9;     // UInt16
#else
    // If header not available, assume integer for wide widths (conservative for pack_untilize)
    // This ensures wide integer tensors get hardware acceleration
    return true;  // Changed to true - prefer pack_untilize block-based path
#endif
}

template <uint32_t cb_id>
constexpr bool is_fp32_format() {
#ifdef UNPACK_DATA_FORMAT_AVAILABLE
    constexpr uint32_t format = unpack_dst_format[cb_id];
    return format == 0;     // Float32
#else
    return false;
#endif
}

// =============================================================================
// Block Splitting Helper Implementation
// =============================================================================

constexpr uint32_t compute_num_columns(uint32_t total_width, uint32_t max_block_width) {
    for (uint32_t block_width = max_block_width; block_width >= 1; --block_width) {
        if (total_width % block_width == 0) {
            return total_width / block_width;
        }
    }
    return total_width;  // fallback: 1 tile per block
}

// =============================================================================
// Unified Init/Uninit Function Implementations
// =============================================================================

template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb>
ALWI void untilize_init() {
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool should_pack_untilize = is_integer_format<input_cb>() || is_fp32_format<input_cb>();

    if constexpr (block_width_tiles > dest_limit && !should_pack_untilize) {
        // Wide width with format that supports standard untilize (e.g., bfloat16)
        ::untilize_init(input_cb);
    } else if constexpr (block_width_tiles > dest_limit) {
        // Wide width with integer/FP32 - use block-based pack_untilize
        constexpr uint32_t num_columns = compute_num_columns(block_width_tiles, dest_limit);
        constexpr uint32_t column_width = block_width_tiles / num_columns;
        pack_untilize_init<column_width, block_width_tiles>(input_cb, output_cb);
    } else {
        // Narrow width - use pack_untilize
        pack_untilize_init<block_width_tiles, block_width_tiles>(input_cb, output_cb);
    }
}

template <uint32_t block_width_tiles, uint32_t input_cb, uint32_t output_cb>
ALWI void untilize_uninit() {
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool should_pack_untilize = is_integer_format<input_cb>() || is_fp32_format<input_cb>();

    if constexpr (block_width_tiles > dest_limit && !should_pack_untilize) {
        // Wide width with format that supports standard untilize
        ::untilize_uninit(input_cb);
    } else {
        // Pack untilize path (for narrow widths or wide integer/FP32)
        pack_untilize_uninit(output_cb);
    }
}

// =============================================================================
// Single Unified Untilize Function Implementation
// =============================================================================

template <
    uint32_t block_width_tiles,
    uint32_t input_cb,
    uint32_t output_cb,
    InitUninitMode init_uninit_mode,
    WaitMode wait_mode,
    uint32_t reconfig_from_cb>
ALWI void untilize(uint32_t num_blocks) {
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool should_pack_untilize = is_integer_format<input_cb>() || is_fp32_format<input_cb>();
    constexpr bool do_init =
        (init_uninit_mode == InitUninitMode::InitAndUninit || init_uninit_mode == InitUninitMode::InitOnly);
    constexpr bool do_uninit =
        (init_uninit_mode == InitUninitMode::InitAndUninit || init_uninit_mode == InitUninitMode::UninitOnly);

    // For wide tensors: use standard untilize only for formats that support it (not integer/FP32)
    if constexpr (block_width_tiles > dest_limit && !should_pack_untilize) {
        // =================================================================
        // STANDARD UNTILIZE PATH
        // Used for wide tensors with formats that support standard untilize (e.g., bfloat16)
        // =================================================================

        if constexpr (do_init) {
            ::untilize_init(input_cb);
        }

        if constexpr (wait_mode == WaitMode::WaitUpfront) {
            uint32_t total_tiles = block_width_tiles * num_blocks;
            cb_wait_front(input_cb, total_tiles);
        }

        for (uint32_t r = 0; r < num_blocks; ++r) {
            if constexpr (wait_mode == WaitMode::Wait) {
                cb_wait_front(input_cb, block_width_tiles);
            }
            cb_reserve_back(output_cb, block_width_tiles);
            untilize_block(input_cb, block_width_tiles, output_cb);
            cb_push_back(output_cb, block_width_tiles);
            cb_pop_front(input_cb, block_width_tiles);
        }

        if constexpr (do_uninit) {
            ::untilize_uninit(input_cb);
        }

    } else if constexpr (block_width_tiles > dest_limit) {
        // =================================================================
        // BLOCK-BASED PACK UNTILIZE PATH
        // Used for wide tensors with integer/FP32 formats
        // Splits wide rows into multiple column chunks that each fit in DEST
        // =================================================================

        constexpr uint32_t num_columns = compute_num_columns(block_width_tiles, dest_limit);
        constexpr uint32_t column_width = block_width_tiles / num_columns;

        if constexpr (do_init) {
            pack_untilize_init<column_width, block_width_tiles>(input_cb, output_cb);
        }

        if constexpr (wait_mode == WaitMode::WaitUpfront) {
            uint32_t total_tiles = block_width_tiles * num_blocks;
            cb_wait_front(input_cb, total_tiles);
        }

        for (uint32_t r = 0; r < num_blocks; ++r) {
            cb_reserve_back(output_cb, block_width_tiles);
            for (uint32_t c = 0; c < num_columns; ++c) {
                if constexpr (wait_mode == WaitMode::Wait) {
                    cb_wait_front(input_cb, column_width);
                }
                pack_untilize_block<column_width, block_width_tiles>(input_cb, 1, output_cb, c);
                cb_pop_front(input_cb, column_width);
            }
            cb_push_back(output_cb, block_width_tiles);
        }

        if constexpr (do_uninit) {
            pack_untilize_uninit(output_cb);
        }

    } else {
        // =================================================================
        // PACK UNTILIZE PATH (SINGLE-PASS)
        // Used when width fits in DEST (optimal for all data types)
        // =================================================================

        if constexpr (do_init) {
            pack_untilize_init<block_width_tiles, block_width_tiles>(input_cb, output_cb);
        }

        if constexpr (wait_mode == WaitMode::WaitUpfront) {
            uint32_t total_tiles = block_width_tiles * num_blocks;
            cb_wait_front(input_cb, total_tiles);
        }

        for (uint32_t r = 0; r < num_blocks; ++r) {
            if constexpr (wait_mode == WaitMode::Wait) {
                cb_wait_front(input_cb, block_width_tiles);
            }
            cb_reserve_back(output_cb, block_width_tiles);
            pack_untilize_block<block_width_tiles, block_width_tiles>(input_cb, 1, output_cb, 0);
            cb_pop_front(input_cb, block_width_tiles);
            cb_push_back(output_cb, block_width_tiles);
        }

        if constexpr (do_uninit) {
            pack_untilize_uninit(output_cb);
        }
    }
}

}  // namespace compute_kernel_lib
