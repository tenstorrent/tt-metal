// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand writer (BRISC / NoC1) — block operation `write_out_block`.
//
// Per block (image group ig, row group rg, column group cg), per real B tile (image n, column w):
// wait 1 page, write it to the this_block_tile_rows out pages (n*Ht + h)*Wt + w of the row group
// straight from L1, flush, pop. Pad pages of a ragged block are waited/popped unwritten.
// One write barrier at kernel end (BB-03).
//
// Raw NoC page write: no kernel_lib helper writes interleaved tile pages from one source page.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_bcast_row_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();

    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t first_block = get_arg_val<uint32_t>(arg++);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(arg++);
    const uint32_t tensor_tile_cols = get_arg_val<uint32_t>(arg++);
    const uint32_t tensor_out_tile_rows = get_arg_val<uint32_t>(arg++);
    const uint32_t row_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t col_groups = get_arg_val<uint32_t>(arg++);
    const uint32_t block_tile_cols = get_arg_val<uint32_t>(arg++);
    const uint32_t block_tile_rows = get_arg_val<uint32_t>(arg++);
    const uint32_t block_images = get_arg_val<uint32_t>(arg++);
    const uint32_t tensor_images = get_arg_val<uint32_t>(arg++);

    const auto out_accessor = TensorAccessor(out_args, out_addr, tile_bytes);

    const uint32_t block_pages = block_images * block_tile_cols;

    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t b = first_block + block_idx;
        const uint32_t cg = b % col_groups;
        const uint32_t rg = (b / col_groups) % row_groups;
        const uint32_t ig = b / (col_groups * row_groups);
        const uint32_t n0 = ig * block_images;
        const uint32_t c0 = cg * block_tile_cols;
        const uint32_t h0 = rg * block_tile_rows;
        const uint32_t this_images = (tensor_images - n0) < block_images ? (tensor_images - n0) : block_images;
        const uint32_t this_cols =
            (tensor_tile_cols - c0) < block_tile_cols ? (tensor_tile_cols - c0) : block_tile_cols;
        const uint32_t this_rows =
            (tensor_out_tile_rows - h0) < block_tile_rows ? (tensor_out_tile_rows - h0) : block_tile_rows;
        const uint32_t real_pages = this_images * this_cols;

        for (uint32_t i = 0; i < this_images; ++i) {
            const uint32_t n = n0 + i;
            for (uint32_t c = 0; c < this_cols; ++c) {
                {
                    DeviceZoneScopedN("wait_bcast_tile");
                    cb_wait_front(cb_bcast_row_tiles, 1);
                }
                const uint32_t src_l1 = get_read_ptr(cb_bcast_row_tiles);
                {
                    DeviceZoneScopedN("write_out");
                    uint32_t out_page = (n * tensor_out_tile_rows + h0) * tensor_tile_cols + c0 + c;
                    for (uint32_t h = 0; h < this_rows; ++h) {
                        noc_async_write_page(out_page, out_accessor, src_l1);
                        out_page += tensor_tile_cols;
                    }
                    noc_async_writes_flushed();
                }
                cb_pop_front(cb_bcast_row_tiles, 1);
            }
        }
        if (real_pages < block_pages) {
            cb_wait_front(cb_bcast_row_tiles, block_pages - real_pages);
            cb_pop_front(cb_bcast_row_tiles, block_pages - real_pages);
        }
    }

    {
        DeviceZoneScopedN("drain_writes");
        noc_async_write_barrier();
    }
}
