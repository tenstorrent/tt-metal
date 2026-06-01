// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t post_out_addr = get_arg_val<uint32_t>(0);
    const uint32_t collapsed_addr = get_arg_val<uint32_t>(1);
    const uint32_t d_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_post_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_collapsed = get_compile_time_arg_val(1);

    constexpr auto post_out_args = TensorAccessorArgs<2>();
    constexpr auto collapsed_args = TensorAccessorArgs<post_out_args.next_compile_time_args_offset()>();

    const auto post_out = TensorAccessor(post_out_args, post_out_addr);
    const auto collapsed = TensorAccessor(collapsed_args, collapsed_addr);

    Noc noc;
    CircularBuffer cb_post(cb_post_out);
    CircularBuffer cb_col(cb_collapsed);

    constexpr uint32_t one_tile = 1;

    // post [1,1,1,H] -> single tile.
    cb_post.wait_front(one_tile);
    noc.async_write(cb_post, post_out, cb_post.get_tile_size(), {.offset_bytes = 0}, {.page_id = 0});

    // collapsed [1,1,1,D] -> d_tiles tiles along the width.
    const uint32_t col_tile_size = cb_col.get_tile_size();
    cb_col.wait_front(d_tiles);
    for (uint32_t n = 0; n < d_tiles; ++n) {
        noc.async_write(cb_col, collapsed, col_tile_size, {.offset_bytes = n * col_tile_size}, {.page_id = n});
    }

    noc.async_write_barrier();

    cb_post.pop_front(one_tile);
    cb_col.pop_front(d_tiles);
}
