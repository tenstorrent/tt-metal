// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// g6_expand writer (BRISC, NoC1), every core of the rectangle: store_block. Each expanded tile E in
// the block is written, as a whole page, once to every out tile row this core owns.
// Out page of (src page s, out tile row r): img * Hc * Wt + r * Wt + wt, with s = img * Wt + wt.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_expanded_tiles_id = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t block_src_col_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_out_row_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_out_row_tiles = get_compile_time_arg_val(5);
    constexpr auto out_args = TensorAccessorArgs<6>();
    constexpr uint32_t image_out_pages = tensor_out_row_tiles * tensor_w_tiles;

    const uint32_t src_start = get_arg_val<uint32_t>(0);
    const uint32_t src_count = get_arg_val<uint32_t>(1);
    const uint32_t row_start = get_arg_val<uint32_t>(2);
    const uint32_t row_count = get_arg_val<uint32_t>(3);
    const uint32_t num_src_blocks = (src_count + block_src_col_tiles - 1) / block_src_col_tiles;

    Noc noc;
    CircularBuffer cb_expanded_tiles(cb_expanded_tiles_id);
    const auto out_accessor = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0));

    for (uint32_t block_idx = 0; block_idx < num_src_blocks; ++block_idx) {
        const uint32_t first = block_idx * block_src_col_tiles;
        const uint32_t extent = src_count - first < block_src_col_tiles ? src_count - first : block_src_col_tiles;
        {
            DeviceZoneScopedN("wait_block");
            cb_expanded_tiles.wait_front(block_src_col_tiles);
        }
        {
            DeviceZoneScopedN("store_block");
            for (uint32_t row_block = 0; row_block < row_count; row_block += block_out_row_tiles) {
                const uint32_t rows =
                    row_count - row_block < block_out_row_tiles ? row_count - row_block : block_out_row_tiles;
                for (uint32_t r = row_start + row_block; r < row_start + row_block + rows; ++r) {
                    // Out-row outer, src inner: consecutive out pages round-robin over the DRAM banks.
                    // A block may span an image boundary, so (img, wt) advance per src page.
                    uint32_t s = src_start + first;
                    uint32_t img = s / tensor_w_tiles;
                    uint32_t wt = s - img * tensor_w_tiles;
                    for (uint32_t t = 0; t < extent; ++t) {
                        noc.async_write(
                            cb_expanded_tiles,
                            out_accessor,
                            page_bytes,
                            {.offset_bytes = t * page_bytes},
                            {.page_id = img * image_out_pages + r * tensor_w_tiles + wt});
                        if (++wt == tensor_w_tiles) {
                            wt = 0;
                            ++img;
                        }
                    }
                }
            }
        }
        {
            DeviceZoneScopedN("write_drain");
            // E must have been read out of L1 before its slot is released to the next block.
            noc.async_writes_flushed();
            cb_expanded_tiles.pop_front(block_src_col_tiles);
        }
    }
    {
        DeviceZoneScopedN("write_drain");
        noc.async_write_barrier();
    }
}
