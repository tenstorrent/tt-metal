// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax Writer Kernel (dim=-2, height reduction)
// Writes output tiles in chunked column order back to row-major DRAM positions.
// Mirrors the reader_h tile ordering: for each chunk, row-by-row across chunk columns.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(0);
    constexpr auto tensor_args = TensorAccessorArgs<1>();

    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t curr_col_in_batch = get_arg_val<uint32_t>(3);
    const uint32_t num_cols = get_arg_val<uint32_t>(4);
    const uint32_t Wt = get_arg_val<uint32_t>(5);
    const uint32_t Ht = get_arg_val<uint32_t>(6);
    const uint32_t chunk_size = get_arg_val<uint32_t>(7);

    if (num_pages == 0) {
        return;
    }

    const uint32_t page_size = get_tile_size(output_cb_index);
    const auto accessor = TensorAccessor(tensor_args, dst_addr, page_size);

    // Write tiles in same chunked column order as reader/compute produced them.
    // Compute tile_id = batch_base + row * Wt + (global_col + col_in_chunk)
    uint32_t global_col = curr_col_in_batch;
    uint32_t batch_base_tile = start_id - curr_col_in_batch;
    uint32_t cols_remaining = num_cols;

    while (cols_remaining > 0) {
        uint32_t cols_left_in_batch = Wt - global_col;
        uint32_t current_chunk = (cols_remaining < chunk_size) ? cols_remaining : chunk_size;
        if (current_chunk > cols_left_in_batch) {
            current_chunk = cols_left_in_batch;
        }

        // Write Ht * current_chunk tiles
        for (uint32_t r = 0; r < Ht; ++r) {
            for (uint32_t c = 0; c < current_chunk; ++c) {
                uint32_t tile_id = batch_base_tile + r * Wt + (global_col + c);
                cb_wait_front(output_cb_index, 1);
                uint32_t l1_read_addr = get_read_ptr(output_cb_index);
                noc_async_write_tile(tile_id, accessor, l1_read_addr);
                noc_async_write_barrier();
                cb_pop_front(output_cb_index, 1);
            }
        }

        global_col += current_chunk;
        cols_remaining -= current_chunk;
        if (global_col >= Wt) {
            global_col = 0;
            batch_base_tile += Ht * Wt;
        }
    }
}
