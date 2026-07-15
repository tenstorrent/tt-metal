// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel writes tiles from the output buffer to interleaved dram.
 */

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(1);
    constexpr uint32_t block_size = get_compile_time_arg_val(2);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_heads = get_compile_time_arg_val(4);
    // Row-tiles per batch element (= sequence_length / TILE_HEIGHT). The output is
    // [batch, num_heads, seq, head_dim]; leading (batch) dims are folded into tile_row.
    constexpr uint32_t rows_per_batch = get_compile_time_arg_val(5);

    constexpr auto output_args = TensorAccessorArgs<6>();
    const uint32_t tile_bytes = get_tile_size(output_cb);

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(1);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(2);

    const auto output_accessor = TensorAccessor(output_args, output_addr);

    // Stride between heads (within one batch element) and between batch elements.
    constexpr uint32_t head_stride = rows_per_batch * head_dim_tiles;
    constexpr uint32_t batch_stride = num_heads * head_stride;

    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        uint32_t head_idx = 0;
        uint32_t tile_idx_in_head = 0;

        // Decompose the folded row index into (batch element, seq row-tile) so each
        // batch element writes into its own [num_heads, seq, head_dim] output slice.
        const uint32_t batch_idx = tile_row / rows_per_batch;
        const uint32_t seq_tile = tile_row % rows_per_batch;
        const uint32_t batch_base = batch_idx * batch_stride;

        uint32_t tile_id = batch_base + seq_tile * head_dim_tiles;

        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            cb_wait_front(output_cb, block_size);
            uint32_t output_read_ptr = get_read_ptr(output_cb);
            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                noc_async_write_page(tile_id, output_accessor, output_read_ptr);
                output_read_ptr += tile_bytes;
                tile_id++;
                tile_idx_in_head++;
                if (tile_idx_in_head == head_dim_tiles) {
                    tile_idx_in_head = 0;
                    head_idx++;
                    tile_id = batch_base + head_idx * head_stride + seq_tile * head_dim_tiles;
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(output_cb, block_size);
        }
    }
    noc_async_write_barrier();
}
