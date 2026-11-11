// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "collapsed_row_scatter.h"

// The same writer is compiled for three roles:
//   0: collapse cores (ROW_MAJOR streams only) -- the first half of the collapsed output tiles (see
//      scatter_collapsed_row); the reader writes the second.
//   1: post core -- post (and pre, given pre_mix) are already whole tiles.
//   2: comb core -- comb is a whole tile.
void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out2 = get_compile_time_arg_val(2);
    constexpr bool has_out2 = get_compile_time_arg_val(3) != 0;
    // Role 0: the collapsed row's 32-column tiles and the h x 32 chunks the compute packs them in.
    constexpr uint32_t out_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t row_chunks = get_compile_time_arg_val(5);

    constexpr auto out_args = TensorAccessorArgs<6>();
    constexpr auto out2_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t out2_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t one_tile = 1;
    Noc noc;

    if constexpr (role == 0) {
        // cb_out2 is the collapsed output shard itself (globally allocated), out_addr its L1 address;
        // the reader writes the second half.
        scatter_collapsed_row(cb_out, row_chunks, cb_out2, out_addr, 0, out_tiles / 2);
    } else {
        const auto out = TensorAccessor(out_args, out_addr);
        CircularBuffer cb(cb_out);
        cb.wait_front(one_tile);
        noc.async_write(cb, out, cb.get_tile_size(), {.offset_bytes = 0}, {.page_id = 0});
        if constexpr (has_out2) {
            const auto out2 = TensorAccessor(out2_args, out2_addr);
            CircularBuffer cb2(cb_out2);
            cb2.wait_front(one_tile);
            noc.async_write(cb2, out2, cb2.get_tile_size(), {.offset_bytes = 0}, {.page_id = 0});
        }
        noc.async_write_barrier();
        cb.pop_front(one_tile);
        if constexpr (has_out2) {
            CircularBuffer(cb_out2).pop_front(one_tile);
        }
    }
}
