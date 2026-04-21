// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Minimal N-input interleaved DRAM reader for kernel_lib tests.
// NUM_INPUTS is a preprocessor define supplied in the KernelDescriptor (1/2/3).
// Runtime args:
//   [0] src_addr_0
//   [1] src_addr_1 (ignored when NUM_INPUTS < 2)
//   [2] src_addr_2 (ignored when NUM_INPUTS < 3)
//   [3] num_tiles
//   [4] start_id
// Compile-time args: [TensorAccessorArgs for each enabled input, in order].

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

#ifndef NUM_INPUTS
#error "reader_n_input.cpp requires -DNUM_INPUTS=1|2|3"
#endif

void kernel_main() {
    const uint32_t src0_addr = get_arg_val<uint32_t>(0);
#if NUM_INPUTS >= 2
    const uint32_t src1_addr = get_arg_val<uint32_t>(1);
#endif
#if NUM_INPUTS >= 3
    const uint32_t src2_addr = get_arg_val<uint32_t>(2);
#endif
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_0 = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr auto s0_args = TensorAccessorArgs<0>();
    experimental::Noc noc;
    experimental::CircularBuffer cb0(cb_id_0);
    const uint32_t bytes0 = get_tile_size(cb_id_0);
    const auto acc0 = TensorAccessor(s0_args, src0_addr, bytes0);

#if NUM_INPUTS >= 2
    constexpr uint32_t cb_id_1 = static_cast<uint32_t>(tt::CBIndex::c_1);
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    experimental::CircularBuffer cb1(cb_id_1);
    const uint32_t bytes1 = get_tile_size(cb_id_1);
    const auto acc1 = TensorAccessor(s1_args, src1_addr, bytes1);
#endif

#if NUM_INPUTS >= 3
    constexpr uint32_t cb_id_2 = static_cast<uint32_t>(tt::CBIndex::c_2);
    constexpr auto s2_args = TensorAccessorArgs<s1_args.next_compile_time_args_offset()>();
    experimental::CircularBuffer cb2(cb_id_2);
    const uint32_t bytes2 = get_tile_size(cb_id_2);
    const auto acc2 = TensorAccessor(s2_args, src2_addr, bytes2);
#endif

    constexpr uint32_t onetile = 1;
    for (uint32_t t = start_id; t < start_id + num_tiles; ++t) {
        cb0.reserve_back(onetile);
        noc.async_read(acc0, cb0, bytes0, {.page_id = t}, {.offset_bytes = 0});
#if NUM_INPUTS >= 2
        cb1.reserve_back(onetile);
        noc.async_read(acc1, cb1, bytes1, {.page_id = t}, {.offset_bytes = 0});
#endif
#if NUM_INPUTS >= 3
        cb2.reserve_back(onetile);
        noc.async_read(acc2, cb2, bytes2, {.page_id = t}, {.offset_bytes = 0});
#endif
        noc.async_read_barrier();
        cb0.push_back(onetile);
#if NUM_INPUTS >= 2
        cb1.push_back(onetile);
#endif
#if NUM_INPUTS >= 3
        cb2.push_back(onetile);
#endif
    }
}
