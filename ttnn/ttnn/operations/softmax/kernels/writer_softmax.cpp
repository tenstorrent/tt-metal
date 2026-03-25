// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax - Writer Kernel
// Writes output tiles from cb_out to DRAM via TensorAccessor.
// dim=-1: row-major write order (Wt tiles per slice)
// dim=-2: column-major write order (Ht tiles per slice, stride Wt)

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = 16;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t HtWt = get_compile_time_arg_val(2);
    constexpr uint32_t is_dim_w = get_compile_time_arg_val(3);
    constexpr auto output_accessor_args = TensorAccessorArgs<4>();

    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_slices = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    // Page size from CB
    constexpr uint32_t page_size = get_tile_size(cb_out);

    // Set up TensorAccessor for output writes
    const auto output_accessor = TensorAccessor(output_accessor_args, dst_addr, page_size);

    // Slice size depends on reduction dimension
    constexpr uint32_t slice_tiles = is_dim_w ? Wt : Ht;

    for (uint32_t slice = 0; slice < num_slices; ++slice) {
        uint32_t slice_id = start_id + slice;

        for (uint32_t t = 0; t < slice_tiles; ++t) {
            // Wait for one tile at a time (cb_out is double-buffered with 2 pages)
            cb_wait_front(cb_out, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_out);

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

            uint64_t noc_addr = output_accessor.get_noc_addr(tile_id);
            noc_async_write(l1_read_addr, noc_addr, page_size);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
