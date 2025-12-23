// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

/**
 * Reader kernel for tilize_untilize operation.
 * Reads row-major sticks (32 rows at a time) from DRAM into CB_in (c_0).
 *
 * CB_in receives row-major data for one tile block (32 rows x width).
 * The compute kernel will tilize this into num_tiles_per_row tiled format tiles.
 *
 * Based on: reader_unary_stick_layout_split_rows_interleaved.cpp pattern
 */

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto src_tensor_args = TensorAccessorArgs<1>();

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // CB setup
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t tile_height = 32;

    // Calculate num_tiles_per_row from stick_size
    // Each tile has TILE_WIDTH=32 elements, and we have BFLOAT16=2 bytes per element
    // stick_size = width * element_size = num_tiles_per_row * TILE_WIDTH * element_size
    // For tilize, CB_in holds row-major data for num_tiles_per_row tiles worth of input
    constexpr uint32_t tile_width = 32;
    constexpr uint32_t element_size = 2;  // BF16
    constexpr uint32_t num_tiles_per_row = stick_size / (tile_width * element_size);

    // Setup tensor accessor
    const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);

    // Array to hold NOC addresses for 32 rows at a time
    uint64_t base_src_noc_addr[tile_height];

    // Process all tile blocks (each block = 32 rows)
    uint32_t stick_id = start_stick_id;
    uint32_t num_blocks = num_sticks / tile_height;

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Get base addresses for 32 consecutive rows
        for (uint32_t row = 0; row < tile_height; row++) {
            base_src_noc_addr[row] = get_noc_addr(stick_id + row, s);
        }

        // Reserve space in CB for num_tiles_per_row tiles
        // This reserves space for 32 rows * width in tiles * tile_size
        cb_reserve_back(cb_id_in0, num_tiles_per_row);

        // Get write pointer
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

        // Read 32 rows (full stick width each)
        for (uint32_t row = 0; row < tile_height; row++) {
            noc_async_read(base_src_noc_addr[row], l1_write_addr, stick_size);
            l1_write_addr += stick_size;
        }

        // Wait for all reads to complete
        noc_async_read_barrier();

        // Push num_tiles_per_row tiles to compute (tilize produces num_tiles_per_row tiles)
        cb_push_back(cb_id_in0, num_tiles_per_row);

        // Advance to next block of 32 sticks
        stick_id += tile_height;
    }
}
