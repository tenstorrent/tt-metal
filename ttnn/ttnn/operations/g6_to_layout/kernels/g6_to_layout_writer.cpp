// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_to_layout writer (BRISC / NoC1): store_live_rows.
//
// Same block walk as the reader. Per block: one cb_wait_front of the nominal block_tiles pages,
// one NoC write per live row of the block's live column bytes, one noc_async_writes_flushed,
// one cb_pop_front. Raw NoC: write_sticks_after_untilize derives its CB row stride and pop count
// from the written bytes, which breaks on ragged column blocks and on blocks spanning images.
//
// Output rows are addressed only through get_noc_addr(page, offset): the RM page stride is
// round_up(W * 2, alignment), not W * 2.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t TILE_DIM = 32;

    constexpr uint32_t cb_rows_out = get_compile_time_arg_val(0);
    constexpr uint32_t block_height_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_tile_rows = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(4);
    constexpr uint32_t image_tile_rows = get_compile_time_arg_val(5);
    constexpr uint32_t height = get_compile_time_arg_val(6);
    constexpr uint32_t width = get_compile_time_arg_val(7);
    constexpr uint32_t col_group_width_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t out_page_stride = get_compile_time_arg_val(10);
    constexpr auto output_args = TensorAccessorArgs<11>();

    constexpr uint32_t datum_bytes = tile_bytes / (TILE_DIM * TILE_DIM);
    constexpr uint32_t tile_row_bytes = TILE_DIM * datum_bytes;
    constexpr uint32_t cb_row_stride = block_width_tiles * tile_row_bytes;
    constexpr uint32_t block_tiles = block_height_tiles * block_width_tiles;
    constexpr uint32_t row_bytes = width * datum_bytes;

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t unit_begin = get_arg_val<uint32_t>(0);
    const uint32_t unit_end = get_arg_val<uint32_t>(1);

    const auto output = TensorAccessor(output_args, output_addr, row_bytes);

    DeviceZoneScopedN("g6_writer");
    uint32_t unit = unit_begin;
    while (unit < unit_end) {
        const uint32_t group = unit / tensor_tile_rows;
        const uint32_t first_tile_row = unit - group * tensor_tile_rows;
        uint32_t rows = block_height_tiles;
        if (tensor_tile_rows - first_tile_row < rows) {
            rows = tensor_tile_rows - first_tile_row;
        }
        if (unit_end - unit < rows) {
            rows = unit_end - unit;
        }
        const uint32_t col_begin = group * col_group_width_tiles;
        const uint32_t col_end =
            col_begin + col_group_width_tiles < tensor_tile_cols ? col_begin + col_group_width_tiles : tensor_tile_cols;

        for (uint32_t col = col_begin; col < col_end; col += block_width_tiles) {
            const uint32_t col_offset = col * tile_row_bytes;
            // A row's alignment padding is part of its own page slot, so the last segment of a
            // row fills the slot: whole DRAM bursts instead of byte-masked partial writes.
            const uint32_t bytes_left = out_page_stride - col_offset;
            const uint32_t block_bytes =
                (col_end - col < block_width_tiles ? col_end - col : block_width_tiles) * tile_row_bytes;
            const uint32_t segment_bytes = bytes_left < block_bytes ? bytes_left : block_bytes;

            cb_wait_front(cb_rows_out, block_tiles);
            const uint32_t block_base = get_read_ptr(cb_rows_out);
            uint32_t image = first_tile_row / image_tile_rows;
            uint32_t ht = first_tile_row - image * image_tile_rows;
            for (uint32_t t = 0; t < rows; ++t) {
                const uint32_t rows_left = height - ht * TILE_DIM;
                const uint32_t live_rows = rows_left < TILE_DIM ? rows_left : TILE_DIM;
                const uint32_t first_page = image * height + ht * TILE_DIM;
                uint32_t src = block_base + t * block_width_tiles * tile_bytes;
                for (uint32_t r = 0; r < live_rows; ++r, src += cb_row_stride) {
                    noc_async_write(src, output.get_noc_addr(first_page + r, col_offset), segment_bytes);
                }
                if (++ht == image_tile_rows) {
                    ht = 0;
                    ++image;
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_rows_out, block_tiles);
        }
        unit += rows;
    }
    noc_async_write_barrier();
}
