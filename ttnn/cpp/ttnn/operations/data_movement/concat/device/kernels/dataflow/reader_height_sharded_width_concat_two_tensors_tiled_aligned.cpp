// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Height-sharded width concat of two TILE tensors whose widths are whole tiles (groups == 1): output tile row i is
// [in0 tiles of row i | in1 tiles of row i], so every row is two contiguous L1 copies. No transpose, no untilize,
// any tile format (the tile size in bytes comes from the data format). The two dataflow RISCs take alternate rows.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t in0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in1_cb = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb = get_compile_time_arg_val(2);
    constexpr uint32_t rows = get_compile_time_arg_val(3);       // tile rows per shard
    constexpr uint32_t w0_bytes = get_compile_time_arg_val(4);   // in0 tiles per row x tile bytes
    constexpr uint32_t w1_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t half = get_compile_time_arg_val(6);       // 0: even rows, 1: odd rows
    constexpr uint32_t out_bytes = w0_bytes + w1_bytes;

    Noc noc;
    const uint32_t in0 = get_read_ptr(in0_cb);
    const uint32_t in1 = get_read_ptr(in1_cb);
    const uint32_t out = get_write_ptr(out_cb);
    const uint32_t nx = my_x[noc.get_noc_id()];
    const uint32_t ny = my_y[noc.get_noc_id()];
    for (uint32_t i = half; i < rows; i += 2) {
        const uint32_t dst = out + i * out_bytes;
        noc.async_read(UnicastEndpoint{}, CoreLocalMem<uint32_t>(dst), w0_bytes, {.noc_x = nx, .noc_y = ny, .addr = in0 + i * w0_bytes}, {.offset_bytes = 0});
        noc.async_read(UnicastEndpoint{}, CoreLocalMem<uint32_t>(dst + w0_bytes), w1_bytes, {.noc_x = nx, .noc_y = ny, .addr = in1 + i * w1_bytes}, {.offset_bytes = 0});
    }
    noc.async_read_barrier();
}
