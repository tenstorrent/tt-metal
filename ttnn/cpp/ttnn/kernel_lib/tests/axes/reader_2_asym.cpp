// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Asymmetric 2-input reader: pushes n0 tiles to CB c_0 and n1 tiles to CB c_1.
// Needed for inter-tile index tests where the broadcast operand has fewer tiles than the
// full operand (Row index: c_1 has Wt tiles; Col index: c_1 has Ht tiles).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src0_addr = get_arg_val<uint32_t>(0);
    const uint32_t src1_addr = get_arg_val<uint32_t>(1);
    const uint32_t n0 = get_arg_val<uint32_t>(2);
    const uint32_t n1 = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb0 = 0;
    constexpr uint32_t cb1 = 1;

    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    const uint32_t bytes0 = get_local_cb_interface(cb0).fifo_page_size;
    const uint32_t bytes1 = get_local_cb_interface(cb1).fifo_page_size;

    const auto s0 = TensorAccessor(src0_args, src0_addr);
    const auto s1 = TensorAccessor(src1_args, src1_addr);

    Noc noc;
    CircularBuffer c0(cb0), c1(cb1);
    constexpr uint32_t onetile = 1;

    // Push c_1 FIRST: it's the held/upfront operand and the chain needs it resident before it drains
    // the streaming c_0. Filling a small c_0 first would block reserve_back before c_1 is pushed -> deadlock.
    for (uint32_t i = 0; i < n1; ++i) {
        c1.reserve_back(onetile);
        noc.async_read(s1, c1, bytes1, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        c1.push_back(onetile);
    }
    for (uint32_t i = 0; i < n0; ++i) {
        c0.reserve_back(onetile);
        noc.async_read(s0, c0, bytes0, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        c0.push_back(onetile);
    }
}
