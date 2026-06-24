// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);

    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    Noc noc;
    CircularBuffer src_cb(src_cb_id);
    uint32_t local_l1_read_addr = input_buffer_address;

    for (uint32_t i = 0; i < num_sticks; ++i) {
        src_cb.reserve_back(1);
        noc.async_read(
            UnicastEndpoint{},
            src_cb,
            stick_size,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
             .noc_y = (uint32_t)my_y[noc.get_noc_id()],
             .addr = local_l1_read_addr},
            {.offset_bytes = 0});
        noc.async_read_barrier();

        src_cb.push_back(1);
        local_l1_read_addr += stick_size;
    }
}
