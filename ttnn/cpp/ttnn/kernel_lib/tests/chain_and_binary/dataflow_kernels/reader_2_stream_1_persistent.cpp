// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Specialised reader: 2 streaming inputs + 1 persistent input.
// - cb_0 (stream A) and cb_1 (stream B): pushed per-tile, num_tiles times
// - cb_2 (persistent): pushed ONCE before the streaming loop
//
// Used by kernels whose compute kernel treats one CB as a persistent tile
// reused across all tiles (e.g. DestReuseMul with a single scale tile).
// Without this separation, a per-tile push to a WaitNoPop CB deadlocks once
// the CB fills.
//
// Runtime args:
//   [0] src_a_addr   (stream, num_tiles pages)
//   [1] src_b_addr   (stream, num_tiles pages)
//   [2] src_pers_addr (persistent, 1 page)
//   [3] num_tiles
//   [4] start_id
// Compile-time args: TensorAccessorArgs for A, then B, then persistent.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    const uint32_t src_a_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_b_addr = get_arg_val<uint32_t>(1);
    const uint32_t src_pers_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_a = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_id_b = static_cast<uint32_t>(tt::CBIndex::c_1);
    constexpr uint32_t cb_id_pers = static_cast<uint32_t>(tt::CBIndex::c_2);

    constexpr auto args_a = TensorAccessorArgs<0>();
    constexpr auto args_b = TensorAccessorArgs<args_a.next_compile_time_args_offset()>();
    constexpr auto args_pers = TensorAccessorArgs<args_b.next_compile_time_args_offset()>();

    experimental::Noc noc;
    experimental::CircularBuffer cb_a(cb_id_a);
    experimental::CircularBuffer cb_b(cb_id_b);
    experimental::CircularBuffer cb_pers(cb_id_pers);

    const uint32_t bytes_a = get_tile_size(cb_id_a);
    const uint32_t bytes_b = get_tile_size(cb_id_b);
    const uint32_t bytes_pers = get_tile_size(cb_id_pers);

    const auto acc_a = TensorAccessor(args_a, src_a_addr, bytes_a);
    const auto acc_b = TensorAccessor(args_b, src_b_addr, bytes_b);
    const auto acc_pers = TensorAccessor(args_pers, src_pers_addr, bytes_pers);

    constexpr uint32_t onetile = 1;

    // Persistent tile — push once before streaming.
    cb_pers.reserve_back(onetile);
    noc.async_read(acc_pers, cb_pers, bytes_pers, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_pers.push_back(onetile);

    for (uint32_t t = start_id; t < start_id + num_tiles; ++t) {
        cb_a.reserve_back(onetile);
        noc.async_read(acc_a, cb_a, bytes_a, {.page_id = t}, {.offset_bytes = 0});
        cb_b.reserve_back(onetile);
        noc.async_read(acc_b, cb_b, bytes_b, {.page_id = t}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_a.push_back(onetile);
        cb_b.push_back(onetile);
    }
}
