// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t dst_dfb_id = get_compile_time_arg_val(0);
    constexpr auto output_args = TensorAccessorArgs<2>();

    DataflowBuffer dst_dfb(dst_dfb_id);
    Noc noc;
    const auto s = TensorAccessor(output_args, output_buffer_address);

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dst_dfb.wait_front(1);
        noc.async_write(dst_dfb, s, stick_size, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dst_dfb.pop_front(1);
    }
}
