// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t output_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t output_stride = get_compile_time_arg_val(2);
    constexpr uint32_t num_input_tensors = get_compile_time_arg_val(3);

    Noc noc;
    DataflowBuffer output_dfb(output_dfb_id);
    const uint32_t base_l1_write_addr = output_dfb.get_write_ptr();

    uint32_t arg_idx = 0;
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        const uint32_t input_num_pages_per_stick = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t input_num_sticks = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t input_write_offset = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t input_read_offset = get_arg_val<uint32_t>(arg_idx++);

        DataflowBuffer input_dfb(input_id);
        uint32_t l1_write_addr = base_l1_write_addr + input_write_offset;
        uint32_t l1_read_addr = input_dfb.get_read_ptr() + input_read_offset;

        noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            page_size,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
             .noc_y = (uint32_t)my_y[noc.get_noc_id()],
             .addr = l1_read_addr});

        for (uint32_t stick_idx = 0; stick_idx < input_num_sticks; stick_idx++) {
            for (uint32_t page_idx = 0; page_idx < input_num_pages_per_stick; page_idx++) {
                CoreLocalMem<uint32_t> dst(l1_write_addr + page_size * page_idx);
                noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    UnicastEndpoint{},
                    dst,
                    page_size,
                    {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                     .addr = l1_read_addr},
                    {.offset_bytes = 0});
                l1_read_addr += page_size;
            }
            l1_write_addr += output_stride;
        }
    }

    noc.async_read_barrier();
}
