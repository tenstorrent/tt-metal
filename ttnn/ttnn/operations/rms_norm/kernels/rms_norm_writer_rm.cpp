// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm ROW_MAJOR writer (untilize-wrapped). Drains cb_rm_out — produced by
// compute's per-chunk untilize — and writes the valid columns of each valid
// stick back to DRAM as row-major sticks. Chunked along W by `reduce_block`
// tiles to mirror the reader / compute pipeline. Native non-aligned handling:
// only `valid_cols` bytes per stick are written (W non-aligned), and only the
// `rows_this_block` valid sticks of the last tile-block (H non-aligned).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_rm_out = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t reduce_block = get_compile_time_arg_val(2);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t W = get_compile_time_arg_val(4);
    constexpr uint32_t out_elem = get_compile_time_arg_val(5);
    constexpr auto output_args = TensorAccessorArgs<6>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t total_sticks = get_arg_val<uint32_t>(3);

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;
    constexpr uint32_t out_tile_row_bytes = TILE_W * out_elem;
    constexpr uint32_t out_padded_chunk_bytes = reduce_block * out_tile_row_bytes;
    constexpr uint32_t chunk_cols = reduce_block * TILE_W;

    // 2-arg TensorAccessor: page size from the tensor's encoded row-major stick size.
    const auto output_accessor = TensorAccessor(output_args, output_addr);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t global_block = start_block + b;
        const uint32_t block_start_stick = global_block * TILE_H;
        uint32_t rows_this_block = total_sticks - block_start_stick;
        if (rows_this_block > TILE_H) {
            rows_this_block = TILE_H;
        }

        for (uint32_t c = 0; c < num_chunks; ++c) {
            const uint32_t col0 = c * chunk_cols;
            uint32_t valid_cols = (col0 < W) ? (W - col0) : 0;
            if (valid_cols > chunk_cols) {
                valid_cols = chunk_cols;
            }
            const uint32_t chunk_row_bytes = valid_cols * out_elem;
            const uint32_t byte_off = col0 * out_elem;

            cb_wait_front(cb_rm_out, reduce_block);
            uint32_t l1 = get_read_ptr(cb_rm_out);
            if (chunk_row_bytes > 0) {
                for (uint32_t r = 0; r < rows_this_block; ++r) {
                    const uint64_t noc_addr = output_accessor.get_noc_addr(block_start_stick + r, byte_off);
                    noc_async_write(l1 + r * out_padded_chunk_bytes, noc_addr, chunk_row_bytes);
                }
                noc_async_write_barrier();
            }
            cb_pop_front(cb_rm_out, reduce_block);
        }
    }
}
