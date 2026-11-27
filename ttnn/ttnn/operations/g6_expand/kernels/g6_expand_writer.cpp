// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand writer (every core): store_block writes each resident E page of the block once per out
// tile row this core owns. E is re-read from L1 for every row, so out rows cost no L1.
//
// Raw Noc API: the only kernel_lib writer (write_sticks_after_untilize) writes row-major sticks; out is
// tiled and written as whole tile pages.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {

constexpr uint32_t cb_expanded_tiles_id = get_compile_time_arg_val(0);
constexpr uint32_t block_src_col_tiles = get_compile_time_arg_val(1);
constexpr uint32_t block_out_row_tiles = get_compile_time_arg_val(2);
constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(4);
constexpr uint32_t tensor_out_row_tiles = get_compile_time_arg_val(5);
constexpr auto out_args = TensorAccessorArgs<6>();

constexpr uint32_t image_out_tiles = tensor_out_row_tiles * tensor_w_tiles;

}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t src_start = get_arg_val<uint32_t>(0);
    const uint32_t src_count = get_arg_val<uint32_t>(1);
    const uint32_t row_start = get_arg_val<uint32_t>(2);
    const uint32_t row_count = get_arg_val<uint32_t>(3);
    const uint32_t row_end = row_start + row_count;

    Noc noc;
    CircularBuffer cb(cb_expanded_tiles_id);
    const auto out_acc = TensorAccessor(out_args, out_addr, page_bytes);

    const uint32_t blocks = (src_count + block_src_col_tiles - 1) / block_src_col_tiles;
    for (uint32_t block_idx = 0; block_idx < blocks; ++block_idx) {
        const uint32_t first = block_idx * block_src_col_tiles;
        const uint32_t extent = (src_count - first < block_src_col_tiles) ? src_count - first : block_src_col_tiles;
        // x page s is out column (s % tensor_w_tiles) of image (s / tensor_w_tiles); a block may span images.
        const uint32_t s0 = src_start + first;
        const uint32_t img0 = s0 / tensor_w_tiles;
        const uint32_t wt0 = s0 - img0 * tensor_w_tiles;
        {
            DeviceZoneScopedN("wait_block");
            cb.wait_front(block_src_col_tiles);
        }
        {
            DeviceZoneScopedN("store_block");
            for (uint32_t r0 = row_start; r0 < row_end; r0 += block_out_row_tiles) {
                const uint32_t r1 = (row_end - r0 < block_out_row_tiles) ? row_end : r0 + block_out_row_tiles;
                // out_row outer, src_col inner: consecutive out pages walk the banks round-robin.
                for (uint32_t r = r0; r < r1; ++r) {
                    uint32_t img = img0;
                    uint32_t wt = wt0;
                    for (uint32_t i = 0; i < extent; ++i) {
                        noc.async_write(
                            cb,
                            out_acc,
                            page_bytes,
                            {.offset_bytes = i * page_bytes},
                            {.page_id = img * image_out_tiles + r * tensor_w_tiles + wt});
                        if (++wt == tensor_w_tiles) {
                            wt = 0;
                            ++img;
                        }
                    }
                }
            }
            noc.async_writes_flushed();
        }
        cb.pop_front(block_src_col_tiles);
    }
    {
        DeviceZoneScopedN("write_drain");
        noc.async_write_barrier();
    }
}
