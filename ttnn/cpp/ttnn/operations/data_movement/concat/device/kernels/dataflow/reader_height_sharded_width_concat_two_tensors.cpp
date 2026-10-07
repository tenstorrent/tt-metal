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
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t input_stick_size_0 = get_arg(args::input_stick_size_0);
    constexpr uint32_t input_stick_size_1 = get_arg(args::input_stick_size_1);
    constexpr uint32_t input_stride_0 = get_arg(args::input_stride_0);
    constexpr uint32_t input_stride_1 = get_arg(args::input_stride_1);

    const uint32_t num_output_pages = get_arg(args::num_output_pages);
    const uint32_t page_start = get_arg(args::page_start);
    const uint32_t page_end = get_arg(args::page_end);
    const uint32_t output_stick_offset = get_arg(args::output_stick_offset);
    const uint32_t input_start_0 = get_arg(args::input_start_0);
    const uint32_t input_start_1 = get_arg(args::input_start_1);

    const uint32_t groups = get_arg(args::groups);

    constexpr uint32_t group_stick_size_0 = input_stick_size_0 / groups;
    constexpr uint32_t group_stick_size_1 = input_stick_size_1 / groups;
    constexpr uint32_t group_stride_0 = input_stride_0 / groups;
    constexpr uint32_t group_stride_1 = input_stride_1 / groups;

    Noc noc;
    // The output and both inputs are borrowed-memory buffers: each one's L1 storage is the shard of
    // the tensor it stands for, which is why the accesses below are raw pointer arithmetic off the
    // buffer's cursor rather than FIFO traffic.
    DataflowBuffer output_dfb(dfb::output);
    DataflowBuffer input_dfb_0(dfb::input_0);
    DataflowBuffer input_dfb_1(dfb::input_1);

    const uint32_t base_l1_write_addr = output_dfb.get_write_ptr();

    uint32_t l1_write_addr_0 = base_l1_write_addr + output_stick_offset;
    const uint32_t l1_read_addr_0 = input_dfb_0.get_read_ptr() + input_start_0;
    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        UnicastEndpoint{},
        group_stick_size_0,
        {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = l1_read_addr_0});

    uint32_t read_offset_0 = l1_read_addr_0;
    uint32_t l1_write_addr_inc_0 = group_stick_size_0 + group_stride_0;
    for (uint32_t page_id_input = page_start; page_id_input < page_end; page_id_input++) {
        for (uint32_t i = 0; i < groups; i++) {
            CoreLocalMem<uint32_t> dst(l1_write_addr_0);
            noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                UnicastEndpoint{},
                dst,
                group_stick_size_0,
                {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                 .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                 .addr = read_offset_0},
                {.offset_bytes = 0});
            l1_write_addr_0 += (l1_write_addr_inc_0);
            read_offset_0 += group_stick_size_0;
        }
    }

    uint32_t l1_write_addr_1 = base_l1_write_addr + output_stick_offset + group_stick_size_0;
    const uint32_t l1_read_addr_1 = input_dfb_1.get_read_ptr() + input_start_1;
    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        UnicastEndpoint{},
        group_stick_size_1,
        {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = l1_read_addr_1});

    uint32_t read_offset_1 = l1_read_addr_1;
    uint32_t l1_write_addr_inc_1 = group_stick_size_1 + group_stride_1;
    for (uint32_t page_id_input = page_start; page_id_input < page_end; page_id_input++) {
        for (uint32_t i = 0; i < groups; i++) {
            CoreLocalMem<uint32_t> dst(l1_write_addr_1);
            noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                UnicastEndpoint{},
                dst,
                group_stick_size_1,
                {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                 .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                 .addr = read_offset_1},
                {.offset_bytes = 0});
            l1_write_addr_1 += (l1_write_addr_inc_1);
            read_offset_1 += group_stick_size_1;
        }
    }

    noc.async_read_barrier();
}
