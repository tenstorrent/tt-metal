// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(0);
    constexpr auto output_args = TensorAccessorArgs<2>();

    experimental::CircularBuffer dst_cb(dst_cb_id);
    experimental::Noc noc;
    const auto s = TensorAccessor(output_args, output_buffer_address);

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dst_cb.wait_front(1);
        noc.async_write(dst_cb, s, stick_size, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dst_cb.pop_front(1);
    }
}
