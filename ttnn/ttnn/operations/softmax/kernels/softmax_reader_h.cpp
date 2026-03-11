// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax Reader Kernel (dim=-2, height reduction)
// Chunked column reads: Ht * chunk_size tiles per chunk, column-interleaved order
// For each chunk: read row-by-row (all chunk cols for row 0, row 1, ... row Ht-1)
// This produces the memory layout expected by reduce<REDUCE_COL> with_row_stride(chunk_size)

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_input = 0;
constexpr uint32_t cb_scaler = 2;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(0);
    constexpr auto tensor_args = TensorAccessorArgs<1>();

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t col_start_tile_id = get_arg_val<uint32_t>(3);
    const uint32_t curr_col_in_batch = get_arg_val<uint32_t>(4);
    const uint32_t num_cols = get_arg_val<uint32_t>(5);
    const uint32_t Wt = get_arg_val<uint32_t>(6);
    const uint32_t Ht = get_arg_val<uint32_t>(7);
    const uint32_t chunk_size = get_arg_val<uint32_t>(8);

    if (num_tiles == 0) {
        return;
    }

    // Prepare reduce scaler tile in c_2 (1.0 for SUM/MAX)
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f);

    // Set up tensor accessor for reading input tiles
    const uint32_t page_size = get_tile_size(cb_input);
    const auto accessor = TensorAccessor(tensor_args, src_addr, page_size);

    // Read tiles in chunked column order for compute kernel compatibility.
    // Compute expects tiles with row_stride = current_chunk:
    //   CB layout: [row0_col0, row0_col1, ..., row0_colC, row1_col0, row1_col1, ...]
    // where C = current_chunk (chunk_size or remainder)
    //
    // In DRAM, tiles are row-major: tile_id = batch_base + row * Wt + col_in_batch

    uint32_t global_col = curr_col_in_batch;
    uint32_t batch_base_tile = col_start_tile_id - curr_col_in_batch;
    uint32_t cols_remaining = num_cols;

    while (cols_remaining > 0) {
        // Determine current chunk size (may be less at end or at batch boundary)
        // Columns left in current batch
        uint32_t cols_left_in_batch = Wt - global_col;
        uint32_t current_chunk = (cols_remaining < chunk_size) ? cols_remaining : chunk_size;
        if (current_chunk > cols_left_in_batch) {
            current_chunk = cols_left_in_batch;
        }

        // Read Ht * current_chunk tiles in row-interleaved order
        for (uint32_t r = 0; r < Ht; ++r) {
            for (uint32_t c = 0; c < current_chunk; ++c) {
                uint32_t tile_id = batch_base_tile + r * Wt + (global_col + c);
                cb_reserve_back(cb_input, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_input);
                noc_async_read_tile(tile_id, accessor, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_input, 1);
            }
        }

        // Advance past this chunk
        global_col += current_chunk;
        cols_remaining -= current_chunk;
        if (global_col >= Wt) {
            // Crossed batch boundary
            global_col = 0;
            batch_base_tile += Ht * Wt;
        }
    }
}
