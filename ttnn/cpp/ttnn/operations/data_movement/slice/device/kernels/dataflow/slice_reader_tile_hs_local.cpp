// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// slice, TILE, height-sharded in and out on the same grid, last dimension only: the selected tile columns of a
// tile row are contiguous in the shard, so every tile row of the output is one local NOC read.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t ht = get_compile_time_arg_val(2);
    constexpr uint32_t wt_in = get_compile_time_arg_val(3);
    constexpr uint32_t wt_out = get_compile_time_arg_val(4);
    constexpr uint32_t ct0 = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t row_in_bytes = wt_in * tile_bytes;
    constexpr uint32_t row_out_bytes = wt_out * tile_bytes;

    cb_reserve_back(cb_out, ht * wt_out);
    uint32_t src = get_read_ptr(cb_in) + ct0 * tile_bytes;
    uint32_t dst = get_write_ptr(cb_out);
    for (uint32_t r = 0; r < ht; ++r) {
        noc_async_read(get_noc_addr(src), dst, row_out_bytes);
        src += row_in_bytes;
        dst += row_out_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_out, ht * wt_out);
}
