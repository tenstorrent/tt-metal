// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

/**
 * Writer kernel for tilize_untilize operation.
 * Writes row-major sticks (32 rows at a time) from CB_out (c_16) to DRAM.
 *
 * CB_out receives row-major data for one tile block (32 rows x width).
 * The untilize operation produces num_tiles_per_row tiles in row-major format.
 *
 * Based on: writer_unary_stick_layout_split_rows_multi_core.cpp pattern
 */

void kernel_main() {
    // Compile-time args
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr auto dst_tensor_args = TensorAccessorArgs<3>();

    // Calculate num_tiles_per_row from output_stick_size
    constexpr uint32_t tile_width = 32;
    constexpr uint32_t element_size = 2;  // BF16
    constexpr uint32_t num_tiles_per_row = output_stick_size / (tile_width * element_size);

    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // Setup tensor accessor
    const auto s = TensorAccessor(dst_tensor_args, dst_addr, output_stick_size);

    // Process all tile blocks (each block = 32 rows)
    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Wait for compute to produce output (num_tiles_per_row tiles from untilize)
        cb_wait_front(cb_id_out, num_tiles_per_row);

        // Get read pointer
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        // Write 32 rows to output
        for (uint32_t row = 0; row < tile_height; row++) {
            uint64_t dst_noc_addr = get_noc_addr(stick_id + row, s);
            noc_async_write(l1_read_addr, dst_noc_addr, output_stick_size);
            l1_read_addr += output_stick_size;
        }

        // Wait for all writes to complete
        noc_async_write_barrier();

        // Pop num_tiles_per_row tiles from CB
        cb_pop_front(cb_id_out, num_tiles_per_row);

        // Advance to next block of 32 sticks
        stick_id += tile_height;
    }
}
