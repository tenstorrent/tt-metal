// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Microbenchmark producer (with gate_up_compute.cpp SE_BENCH SE_BENCH_WAITS): pushes the weight CB once (W_TILES)
// and X_BLOCKS x blocks of X_BLK tiles as fast as the compute frees them, no data movement - the compute's waits are
// satisfied as soon as possible. CT: 0 X_CB, 1 X_BLK, 2 X_BLOCKS, 3 W_CB, 4 W_TILES
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t x_cb = get_compile_time_arg_val(0), x_blk = get_compile_time_arg_val(1);
    constexpr uint32_t x_blocks = get_compile_time_arg_val(2), w_cb = get_compile_time_arg_val(3);
    constexpr uint32_t w_tiles = get_compile_time_arg_val(4);
#ifdef BENCH_FILL  // fill the x and weight CBs once from DRAM tensors (RT 0 x, 1 weights, 2 x tiles; CT 5.. their
                   // accessors)
    {
        constexpr auto xa = TensorAccessorArgs<5>();
        constexpr auto wa = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
        const auto xs = TensorAccessor(xa, get_arg_val<uint32_t>(0), get_tile_size(x_cb));
        const auto ws = TensorAccessor(wa, get_arg_val<uint32_t>(1), get_tile_size(w_cb));
        const uint32_t x_tiles = get_arg_val<uint32_t>(2);  // (the x CB's tiles)
        for (uint32_t i = 0; i < x_tiles; ++i) {
            noc_async_read_tile(i, xs, get_write_ptr(x_cb) + i * get_tile_size(x_cb));
        }
        for (uint32_t i = 0; i < w_tiles; ++i) {
            noc_async_read_tile(i, ws, get_write_ptr(w_cb) + i * get_tile_size(w_cb));
        }
        noc_async_read_barrier();
    }
#endif
    cb_reserve_back(w_cb, w_tiles);
    cb_push_back(w_cb, w_tiles);
    for (uint32_t i = 0; i < x_blocks; ++i) {
        cb_reserve_back(x_cb, x_blk);
        cb_push_back(x_cb, x_blk);
    }
}
