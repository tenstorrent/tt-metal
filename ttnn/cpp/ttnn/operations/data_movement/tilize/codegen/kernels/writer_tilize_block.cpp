// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Block tilize writer: writes a rectangular sub-block of tiles to DRAM.
//
// Each core writes a (block_Ht x block_Wt) region of output tiles.
// Tiles are produced row-major within the block but must be written to
// 2D positions in the output tensor using (start_tile, block_Wt, full_Wt).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile = get_arg_val<uint32_t>(2);  // first output tile ID (row * Wt + col)
    uint32_t block_Wt = get_arg_val<uint32_t>(3);    // tiles per row in this core's block
    uint32_t full_Wt = get_arg_val<uint32_t>(4);     // total tiles per row in the full tensor

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();
    constexpr uint32_t BATCH = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());

    const auto d = TensorAccessor(dst_args, dst_addr, page_size);

    Noc noc;
    CircularBuffer cb_out_buf(cb_out);

    // Compute output tile IDs: row-major within the block, strided in the tensor
    // Block tile (r, c) maps to output tile: start_tile + r * full_Wt + c
    uint32_t tiles_written = 0;
    uint32_t row_tile_id = start_tile;  // first tile of current row
    uint32_t col_in_row = 0;

    if constexpr (BATCH > 1) {
        uint32_t tiles_left = num_tiles;

        // Prime the pipeline
        uint32_t batch = (tiles_left < BATCH) ? tiles_left : BATCH;
        cb_out_buf.wait_front(batch);
        uint32_t l1_offset = 0;
        for (uint32_t t = 0; t < batch; t++) {
            uint32_t tile_id = row_tile_id + col_in_row;
            noc.async_write(
                cb_out_buf, d, page_size, {.offset_bytes = l1_offset}, {.page_id = tile_id, .offset_bytes = 0});
            l1_offset += page_size;
            col_in_row++;
            if (col_in_row == block_Wt) {
                col_in_row = 0;
                row_tile_id += full_Wt;
            }
        }
        tiles_left -= batch;
        uint32_t prev_batch = batch;

        // Steady state
        while (tiles_left > 0) {
            batch = (tiles_left < BATCH) ? tiles_left : BATCH;
            cb_out_buf.wait_front(prev_batch + batch);
            noc.async_writes_flushed();
            cb_out_buf.pop_front(prev_batch);

            // Offsets are relative to the CB read pointer, which pop_front
            // just advanced past the retired batch.
            l1_offset = 0;
            for (uint32_t t = 0; t < batch; t++) {
                uint32_t tile_id = row_tile_id + col_in_row;
                noc.async_write(
                    cb_out_buf, d, page_size, {.offset_bytes = l1_offset}, {.page_id = tile_id, .offset_bytes = 0});
                l1_offset += page_size;
                col_in_row++;
                if (col_in_row == block_Wt) {
                    col_in_row = 0;
                    row_tile_id += full_Wt;
                }
            }
            tiles_left -= batch;
            prev_batch = batch;
        }

        noc.async_writes_flushed();
        cb_out_buf.pop_front(prev_batch);
    } else {
        for (uint32_t i = 0; i < num_tiles; i++) {
            uint32_t tile_id = row_tile_id + col_in_row;
            cb_out_buf.wait_front(1);
            noc.async_write(cb_out_buf, d, page_size, {.offset_bytes = 0}, {.page_id = tile_id, .offset_bytes = 0});
            noc.async_writes_flushed();
            cb_out_buf.pop_front(1);
            col_in_row++;
            if (col_in_row == block_Wt) {
                col_in_row = 0;
                row_tile_id += full_Wt;
            }
        }
    }
    noc.async_write_barrier();
}
