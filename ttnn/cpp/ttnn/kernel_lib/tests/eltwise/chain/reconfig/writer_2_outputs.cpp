// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes num_tiles tiles from CBs c_16, c_17 to 2 output tensors.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t dst0_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst1_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb0 = 16;
    constexpr uint32_t cb1 = 17;

    constexpr auto dst0_args = TensorAccessorArgs<0>();
    constexpr auto dst1_args = TensorAccessorArgs<dst0_args.next_compile_time_args_offset()>();

    const uint32_t bytes0 = get_local_cb_interface(cb0).fifo_page_size;
    const uint32_t bytes1 = get_local_cb_interface(cb1).fifo_page_size;

    const auto s0 = TensorAccessor(dst0_args, dst0_addr);
    const auto s1 = TensorAccessor(dst1_args, dst1_addr);

    Noc noc;
    CircularBuffer c0(cb0), c1(cb1);

    constexpr uint32_t onetile = 1;
    for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
        c0.wait_front(onetile);
        c1.wait_front(onetile);
        noc.async_write(c0, s0, bytes0, {}, {.page_id = i});
        noc.async_write(c1, s1, bytes1, {}, {.page_id = i});
        noc.async_writes_flushed();
        c0.pop_front(onetile);
        c1.pop_front(onetile);
    }
    noc.async_write_barrier();
}
