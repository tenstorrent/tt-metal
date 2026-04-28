// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// 3-input interleaved DRAM reader for eltwise helper tests.
// Compile-time: TensorAccessorArgs blobs for inputs 0, 1, 2 (in order).
// Runtime:      [src0_addr, src1_addr, src2_addr, num_tiles, start_id].

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr auto s0_args = TensorAccessorArgs<0>();
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    constexpr auto s2_args = TensorAccessorArgs<s1_args.next_compile_time_args_offset()>();

    const uint32_t src0_addr = get_arg_val<uint32_t>(0);
    const uint32_t src1_addr = get_arg_val<uint32_t>(1);
    const uint32_t src2_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t start_id = get_arg_val<uint32_t>(4);

    experimental::Noc noc;
    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb1(tt::CBIndex::c_1);
    experimental::CircularBuffer cb2(tt::CBIndex::c_2);
    const uint32_t b0 = get_tile_size(tt::CBIndex::c_0);
    const uint32_t b1 = get_tile_size(tt::CBIndex::c_1);
    const uint32_t b2 = get_tile_size(tt::CBIndex::c_2);
    const auto a0 = TensorAccessor(s0_args, src0_addr, b0);
    const auto a1 = TensorAccessor(s1_args, src1_addr, b1);
    const auto a2 = TensorAccessor(s2_args, src2_addr, b2);

    for (uint32_t t = start_id; t < start_id + num_tiles; ++t) {
        cb0.reserve_back(1);
        noc.async_read(a0, cb0, b0, {.page_id = t}, {.offset_bytes = 0});
        cb1.reserve_back(1);
        noc.async_read(a1, cb1, b1, {.page_id = t}, {.offset_bytes = 0});
        cb2.reserve_back(1);
        noc.async_read(a2, cb2, b2, {.page_id = t}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb0.push_back(1);
        cb1.push_back(1);
        cb2.push_back(1);
    }
}
