// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core = get_arg_val<uint32_t>(1);

    const auto s0 = TensorAccessor(dst_args, dst_addr);

    constexpr uint32_t cb_id_out0 = 24;

    // Create experimental CircularBuffer for Device 2.0 API
    experimental::CircularBuffer cb_out0(cb_id_out0);
    const uint32_t start_id = 0;
    uint32_t i_stick = start_id;
    uint32_t sticks_read = 0;

    for (uint32_t iter = i_stick; iter < num_sticks_per_core; ++iter) {
        cb_out0.wait_front(1);
        uint32_t l1_read_addr = cb_out0.get_read_ptr();
        uint64_t dst_noc_addr = s0.get_noc_addr(iter);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        noc_async_write_barrier();
        cb_out0.pop_front(1);
    }
}
