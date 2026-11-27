// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand writer (every core of the rectangle): store_block.
//
// Each expanded tile E in cb_expanded_tiles is the out tile of its column for every out row, so
// a block of E pages is written once per out row this core owns, straight from L1. Out pages are
// whole 2 KB tiles, so no kernel_lib write helper applies (they write row-major sticks).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

constexpr uint32_t cb_expanded_tiles = get_compile_time_arg_val(0);
constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
constexpr uint32_t block_src_col_tiles = get_compile_time_arg_val(2);
constexpr uint32_t block_out_row_tiles = get_compile_time_arg_val(3);
constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(4);
constexpr uint32_t tensor_out_row_tiles = get_compile_time_arg_val(5);
constexpr auto out_args = TensorAccessorArgs<6>();

constexpr uint32_t out_image_pages = tensor_out_row_tiles * tensor_w_tiles;

}  // namespace

void kernel_main() {
    const uint32_t src_start = get_arg_val<uint32_t>(0);
    const uint32_t src_count = get_arg_val<uint32_t>(1);
    const uint32_t row_start = get_arg_val<uint32_t>(2);
    const uint32_t row_count = get_arg_val<uint32_t>(3);

    Noc noc;
    CircularBuffer cb(cb_expanded_tiles);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0), page_bytes);

    uint32_t block_first_src = src_start;
    for (uint32_t done = 0; done < src_count; done += block_src_col_tiles) {
        const uint32_t extent = src_count - done < block_src_col_tiles ? src_count - done : block_src_col_tiles;
        // A src block may cross an image boundary: track (image, column) per page.
        const uint32_t first_image = block_first_src / tensor_w_tiles;
        const uint32_t first_col = block_first_src - first_image * tensor_w_tiles;
        {
            DeviceZoneScopedN("wait_block");
            cb.wait_front(block_src_col_tiles);
        }
        {
            DeviceZoneScopedN("store_block");
            for (uint32_t rows_done = 0; rows_done < row_count; rows_done += block_out_row_tiles) {
                const uint32_t rows =
                    row_count - rows_done < block_out_row_tiles ? row_count - rows_done : block_out_row_tiles;
                // out_row outer, src_col inner: consecutive out pages, so banks round-robin.
                for (uint32_t r = 0; r < rows; ++r) {
                    const uint32_t row_page = (row_start + rows_done + r) * tensor_w_tiles;
                    uint32_t image = first_image;
                    uint32_t col = first_col;
                    for (uint32_t s = 0; s < extent; ++s) {
                        noc.async_write(
                            cb,
                            out,
                            page_bytes,
                            {.offset_bytes = s * page_bytes},
                            {.page_id = image * out_image_pages + row_page + col});
                        if (++col == tensor_w_tiles) {
                            col = 0;
                            ++image;
                        }
                    }
                }
            }
            // E must stay put until the NoC has read it for every write of this block.
            noc.async_writes_flushed();
        }
        cb.pop_front(block_src_col_tiles);
        block_first_src += block_src_col_tiles;
    }

    {
        DeviceZoneScopedN("write_drain");
        noc.async_write_barrier();
    }
}
