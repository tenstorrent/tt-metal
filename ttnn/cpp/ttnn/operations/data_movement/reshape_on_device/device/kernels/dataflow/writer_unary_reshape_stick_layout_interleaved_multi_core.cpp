// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(1);
    uint32_t num_read_per_barrier = get_arg_val<uint32_t>(2);
    uint32_t num_sticks_per_cb_push = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t new_stick_size = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    const auto s = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    CircularBuffer cb_output(cb_out0);

    uint32_t i_stick = start_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read; ++iter) {
        cb_output.wait_front(num_sticks_per_cb_push);
        uint32_t cb_read_offset = 0;

        for (uint32_t i = 0; i < num_read_per_barrier; ++i) {
            noc.async_write(
                cb_output,
                s,
                new_stick_size,
                {.offset_bytes = cb_read_offset},
                {.page_id = i_stick, .offset_bytes = 0});
            cb_read_offset += new_stick_size;
            i_stick += 1;
        }
        noc.async_write_barrier();
        cb_output.pop_front(num_sticks_per_cb_push);
    }
}
