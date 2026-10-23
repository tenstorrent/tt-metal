// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core = get_arg_val<uint32_t>(1);

    const auto s0 = TensorAccessor(dst_args, dst_addr);

    constexpr uint32_t dfb_id_out0 = 24;

    Noc noc;
    // Create DataflowBuffer for Device 2.0 API
    DataflowBuffer dfb_out0(dfb_id_out0);
    const uint32_t start_id = 0;
    uint32_t i_stick = start_id;
    uint32_t sticks_read = 0;

    for (uint32_t iter = i_stick; iter < num_sticks_per_core; ++iter) {
        dfb_out0.wait_front(1);
        noc.async_write(dfb_out0, s0, page_size, {.offset_bytes = 0}, {.page_id = iter, .offset_bytes = 0});
        noc.async_write_barrier();
        dfb_out0.pop_front(1);
    }
}
