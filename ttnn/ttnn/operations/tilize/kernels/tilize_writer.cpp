// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize writer (BRISC / NoC1) — block operation `store_tiles_block`.
//
// Mirrors the reader's block walk. Per tile-row: wait `width` tiles, issue `width` tile-page
// writes, flush (L1 slot reusable), pop `width`; one write barrier at the end. out_tile_id = r * tensor_cols_tiles + c.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_tiles_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tiles_out_tail = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_cols_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t last_block_width_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_col_blocks = get_compile_time_arg_val(6);
    constexpr bool has_tail = get_compile_time_arg_val(7) != 0;
    constexpr auto out_args = TensorAccessorArgs<8>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start_tile = get_arg_val<uint32_t>(1);
    const uint32_t core_rows_tiles = get_arg_val<uint32_t>(2);
    const uint32_t col_start_tile = get_arg_val<uint32_t>(3);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(4);

    const auto out_acc = TensorAccessor(out_args, dst_addr, out_page_bytes);

    for (uint32_t col_block = 0; col_block < num_col_blocks; ++col_block) {
        const bool is_tail = has_tail && (col_block == num_col_blocks - 1);
        const uint32_t cb_id = is_tail ? cb_tiles_out_tail : cb_tiles_out;
        const uint32_t width_tiles = is_tail ? last_block_width_tiles : block_width_tiles;
        const uint32_t block_col_start = col_start_tile + col_block * block_width_tiles;

        for (uint32_t rb_start = 0; rb_start < core_rows_tiles; rb_start += block_height_tiles) {
            const uint32_t rb_end =
                (rb_start + block_height_tiles < core_rows_tiles) ? rb_start + block_height_tiles : core_rows_tiles;
            for (uint32_t r = rb_start; r < rb_end; ++r) {
                const uint32_t first_tile = (row_start_tile + r) * tensor_cols_tiles + block_col_start;
                cb_wait_front(cb_id, width_tiles);
                uint32_t l1_read_addr = get_read_ptr(cb_id);
                for (uint32_t c = 0; c < width_tiles; ++c) {
                    noc_async_write(l1_read_addr, out_acc.get_noc_addr(first_tile + c), out_page_bytes);
                    l1_read_addr += out_page_bytes;
                }
                // Flushed = data has left L1, so the slot may be recycled; acks are collected once
                // at the end, keeping writes of consecutive tile-rows in flight together.
                noc_async_writes_flushed();
                cb_pop_front(cb_id, width_tiles);
            }
        }
    }
    noc_async_write_barrier();
}
