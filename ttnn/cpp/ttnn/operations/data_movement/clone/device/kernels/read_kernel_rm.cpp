// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(1);
    constexpr auto input_args = TensorAccessorArgs<2>();

    experimental::CircularBuffer src_cb(src_cb_id);
    experimental::Noc noc;
    const auto s = TensorAccessor(input_args, input_buffer_address, input_page_size);

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        src_cb.reserve_back(1);
        noc.async_read(s, src_cb, stick_size, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        src_cb.push_back(1);
    }
}
