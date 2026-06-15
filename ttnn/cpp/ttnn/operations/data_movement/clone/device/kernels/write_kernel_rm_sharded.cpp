// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);

    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(0);
    Noc noc;
    CircularBuffer dst_cb(dst_cb_id);

    uint32_t local_l1_write_addr = output_buffer_address;

    for (uint32_t i = 0; i < num_sticks; ++i) {
        dst_cb.wait_front(1);
        noc.async_write(
            dst_cb,
            UnicastEndpoint{},
            stick_size,
            {.offset_bytes = 0},
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
             .noc_y = (uint32_t)my_y[noc.get_noc_id()],
             .addr = local_l1_write_addr});
        noc.async_write_barrier();

        dst_cb.pop_front(1);
        local_l1_write_addr += stick_size;
    }
}
