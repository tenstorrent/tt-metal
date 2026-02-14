// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "utils/buffer.h"
#include "utils/math.h"

// This kernel is designed for height-sharded tensors
// Each core processes a subset of rows
// Padding is applied to the height dimension if needed

void kernel_main() {
    // Compile-time args
    constexpr uint32_t cb_id_src0 = get_compile_time_arg_val(0);  // Input data CB
    constexpr uint32_t cb_id_src1 = get_compile_time_arg_val(1);  // Tile CB
    constexpr uint32_t cb_id_src2 = get_compile_time_arg_val(2);  // Pad value CB

    // Runtime args
    uint32_t num_input_rows = get_arg_val<uint32_t>(0);           // Number of input rows per core
    uint32_t input_shard_width_bytes = get_arg_val<uint32_t>(1);  // Width of each row in bytes
    uint32_t row_stride_bytes = get_arg_val<uint32_t>(2);         // Stride between batches
    uint32_t ntiles_per_batch = get_arg_val<uint32_t>(3);         // Number of tiles per batch
    uint32_t num_padded_rows = get_arg_val<uint32_t>(4);          // Number of padded rows to add
    uint32_t num_batches = get_arg_val<uint32_t>(5);              // Number of batches
    uint32_t packed_pad_value = get_arg_val<uint32_t>(6);         // Pad value (packed)

    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH_BYTES = 64;  // 16 elements * 4 bytes (assuming float32/bfloat16 packed)

    // Buffer for writing pad values
    uint32_t l1_pad_addr = get_write_ptr(cb_id_src2);
    volatile tt_l1_ptr uint32_t* l1_pad_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_pad_addr);

    // Initialize pad buffer with packed pad value
    uint32_t num_elements_per_row = input_shard_width_bytes / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_elements_per_row; i++) {
        l1_pad_ptr[i] = packed_pad_value;
    }

    uint32_t input_row_offset = 0;

    for (uint32_t batch = 0; batch < num_batches; batch++) {
        uint32_t current_row_in_batch = 0;
        uint32_t tiles_produced = 0;

        // Process input rows for this batch
        while (current_row_in_batch < num_input_rows) {
            // Calculate how many rows to process in this tile
            uint32_t rows_remaining = num_input_rows - current_row_in_batch;
            uint32_t rows_in_this_tile = min(rows_remaining, TILE_HEIGHT);

            // Get write pointer for tile CB
            uint32_t l1_write_addr = get_write_ptr(cb_id_src1);

            // Copy input rows to tile CB
            for (uint32_t row = 0; row < rows_in_this_tile; row++) {
                uint64_t src_noc_addr = get_noc_addr(input_row_offset + current_row_in_batch + row, cb_id_src0);
                noc_async_read(src_noc_addr, l1_write_addr + row * input_shard_width_bytes, input_shard_width_bytes);
            }

            // If we need to pad rows in this tile to reach TILE_HEIGHT
            if (rows_in_this_tile < TILE_HEIGHT && num_padded_rows > 0) {
                uint32_t rows_to_pad = min(TILE_HEIGHT - rows_in_this_tile, num_padded_rows);
                for (uint32_t row = 0; row < rows_to_pad; row++) {
                    noc_async_read(get_noc_addr(0, cb_id_src2), 
                                  l1_write_addr + (rows_in_this_tile + row) * input_shard_width_bytes, 
                                  input_shard_width_bytes);
                }
                if (num_padded_rows >= rows_to_pad) {
                    num_pushed_rows -= rows_to_pad;
                }
            }

            noc_async_read_barrier();
            push_tile(cb_id_src1);

            current_row_in_batch += rows_in_this_tile;
            tiles_produced++;

            if (tiles_produced >= ntiles_per_batch) {
                break;
            }
        }

        // Add any remaining padded tiles for this batch
        while (tiles_produced < ntiles_per_batch && num_padded_rows >= TILE_HEIGHT) {
            uint32_t l1_write_addr = get_write_ptr(cb_id_src1);

            // Fill entire tile with pad values
            for (uint32_t row = 0; row < TILE_HEIGHT; row++) {
                noc_async_read(get_noc_addr(0, cb_id_src2),
                              l1_write_addr + row * input_shard_width_bytes,
                              input_shard_width_bytes);
            }

            noc_async_read_barrier();
            push_tile(cb_id_src1);

            num_padded_rows -= TILE_HEIGHT;
            tiles_produced++;
        }

        input_row_offset += num_input_rows;
    }
}
