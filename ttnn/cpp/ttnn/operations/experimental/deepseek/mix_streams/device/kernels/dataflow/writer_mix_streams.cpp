// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Streams this core's slice of the mixed output back. TILE output is one 32x32 page
// per n-tile. ROW_MAJOR output untilizes the 32x32 dest tile; the first hc rows of
// that untilized page are scattered into the matching RM pages at the 32-wide column
// offset. WIDTH_SHARDED RM pages are one shard-row (page_id = row * KW + col_shard).
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t hc = get_compile_time_arg_val(1);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t out_is_rm = get_compile_time_arg_val(3);
    constexpr uint32_t out_row_page_stride = get_compile_time_arg_val(4);
    constexpr uint32_t tiles_per_page = get_compile_time_arg_val(5);
    constexpr auto out_args = TensorAccessorArgs<6>();

    const auto out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer out_cb(cb_out);

    constexpr uint32_t one_tile = 1;
    constexpr uint32_t tile_w = 32;
    constexpr uint32_t row_bytes = tile_w * sizeof(uint16_t);
    const uint32_t tile_size_bytes = out_cb.get_tile_size();

    for (uint32_t page = start_tile; page < start_tile + num_tiles; ++page) {
        out_cb.wait_front(one_tile);
        if constexpr (out_is_rm) {
            const uint32_t t = page / n_tiles;
            const uint32_t n_idx = page % n_tiles;
            const uint32_t col_shard = n_idx / tiles_per_page;
            const uint32_t col_offset = (n_idx % tiles_per_page) * row_bytes;
            for (uint32_t r = 0; r < hc; ++r) {
                noc.async_write(
                    out_cb,
                    out,
                    row_bytes,
                    {.offset_bytes = r * row_bytes},
                    {.page_id = (t * hc + r) * out_row_page_stride + col_shard, .offset_bytes = col_offset});
            }
        } else {
            noc.async_write(out_cb, out, tile_size_bytes, {.offset_bytes = 0}, {.page_id = page});
        }
        noc.async_write_barrier();
        out_cb.pop_front(one_tile);
    }
}
