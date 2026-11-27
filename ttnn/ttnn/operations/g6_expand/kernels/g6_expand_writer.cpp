// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand writer (BRISC, NoC1), every core of the rectangle.
//
// Per src block: wait for the block's E tiles, write each of them to every out tile row this core owns
// (store_block), flush, pop. Raw Noc API: out is tiled and written as whole pages; no kernel_lib helper
// writes tile pages from a CB.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_expanded_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t block_src_col_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_out_row_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_out_row_tiles = get_compile_time_arg_val(5);
    constexpr auto out_args = TensorAccessorArgs<6>();
    constexpr uint32_t image_out_pages = tensor_out_row_tiles * tensor_w_tiles;

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t src_start = get_arg_val<uint32_t>(0);
    const uint32_t src_count = get_arg_val<uint32_t>(1);
    const uint32_t row_start = get_arg_val<uint32_t>(2);
    const uint32_t row_end = row_start + get_arg_val<uint32_t>(3);

    Noc noc;
    CircularBuffer cb(cb_expanded_tiles);
    const auto out = TensorAccessor(out_args, out_addr, page_bytes);

    for (uint32_t first = 0; first < src_count; first += block_src_col_tiles) {
        const uint32_t extent = src_count - first < block_src_col_tiles ? src_count - first : block_src_col_tiles;
        // x page s is column (s % tensor_w_tiles) of image (s / tensor_w_tiles); a block may cross images.
        const uint32_t block_src = src_start + first;
        const uint32_t block_img = block_src / tensor_w_tiles;
        const uint32_t block_col = block_src - block_img * tensor_w_tiles;
        {
            DeviceZoneScopedN("wait_block");
            cb.wait_front(block_src_col_tiles);
        }
        for (uint32_t row_block = row_start; row_block < row_end; row_block += block_out_row_tiles) {
            DeviceZoneScopedN("store_block");
            const uint32_t rows_end =
                row_end - row_block < block_out_row_tiles ? row_end : row_block + block_out_row_tiles;
            // out_row outer, src_col inner: consecutive out pages, so the writes rotate over the banks.
            for (uint32_t r = row_block; r < rows_end; ++r) {
                uint32_t img_row_base = block_img * image_out_pages + r * tensor_w_tiles;
                uint32_t col = block_col;
                for (uint32_t i = 0; i < extent; ++i) {
                    noc.async_write(
                        cb, out, page_bytes, {.offset_bytes = i * page_bytes}, {.page_id = img_row_base + col});
                    if (++col == tensor_w_tiles) {
                        col = 0;
                        img_row_base += image_out_pages;
                    }
                }
            }
        }
        {
            DeviceZoneScopedN("write_drain");
            noc.async_writes_flushed();
            cb.pop_front(block_src_col_tiles);
        }
    }
    {
        DeviceZoneScopedN("write_drain");
        noc.async_write_barrier();
    }
}
