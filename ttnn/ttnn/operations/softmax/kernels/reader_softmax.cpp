// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax - Reader Kernel
// Reads input tiles from DRAM into cb_input via TensorAccessor.
// Generates reduce scaler into cb_scaler (value = 1.0).
// dim=-1: row-major order (Wt tiles per row)
// dim=-2: column-major order (Ht tiles per column, stride Wt)

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_input = 0;
constexpr uint32_t cb_scaler = 1;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t HtWt = get_compile_time_arg_val(2);
    constexpr uint32_t is_dim_w = get_compile_time_arg_val(3);
    constexpr auto input_accessor_args = TensorAccessorArgs<4>();

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_slices = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    // Page size from CB
    constexpr uint32_t page_size = get_tile_size(cb_input);

    // Set up TensorAccessor for input reads
    const auto input_accessor = TensorAccessor(input_accessor_args, src_addr, page_size);

    // Generate reduce scaler (1.0) into cb_scaler -- done once, persists for all slices
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f);

    // Determine slice size based on dimension
    constexpr uint32_t slice_tiles = is_dim_w ? Wt : Ht;

    for (uint32_t slice = 0; slice < num_slices; ++slice) {
        uint32_t slice_id = start_id + slice;

        // For dim=-1: slice_id is a tile-row index (row_idx in [0, NC*Ht))
        //   tiles in this row: row_idx * Wt + wt, for wt in [0, Wt)
        // For dim=-2: slice_id is a tile-column index (col_idx in [0, NC*Wt))
        //   tiles in this column: (col_idx / Wt) * HtWt + ht * Wt + (col_idx % Wt), for ht in [0, Ht)

        cb_reserve_back(cb_input, slice_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_input);

        for (uint32_t t = 0; t < slice_tiles; ++t) {
            uint32_t tile_id;
            if constexpr (is_dim_w) {
                // Row-major: tile_id = slice_id * Wt + t
                tile_id = slice_id * Wt + t;
            } else {
                // Column-major: tile_id = batch_offset + t * Wt + col_within_batch
                uint32_t batch_idx = slice_id / Wt;
                uint32_t col_within_batch = slice_id % Wt;
                tile_id = batch_idx * HtWt + t * Wt + col_within_batch;
            }

            uint64_t noc_addr = input_accessor.get_noc_addr(tile_id);
            noc_async_read(noc_addr, l1_write_addr, page_size);
            l1_write_addr += page_size;
        }

        noc_async_read_barrier();
        cb_push_back(cb_input, slice_tiles);
    }
}
