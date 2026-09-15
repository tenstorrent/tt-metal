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

// A sharded tensor is laid out as its 2D flattening, so a height concat has to interleave per
// leading index rather than append whole shards (#55342): block b of input i lands
// b * output_block_stride into the output shard and starts b * input_block_stride into the input.
// num_blocks is 1 for width concat and for the rank-4 (1, 1, H, W) case, where the block loop
// runs once and this is exactly the append it always was.
void kernel_main() {
    constexpr uint32_t output_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t output_stride = get_compile_time_arg_val(2);
    constexpr uint32_t num_input_tensors = get_compile_time_arg_val(3);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t output_block_stride = get_compile_time_arg_val(5);

    Noc noc;
    DataflowBuffer output_dfb(output_dfb_id);
    const uint32_t base_l1_write_addr = output_dfb.get_write_ptr();

    uint32_t arg_idx = 0;
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        // input_num_sticks is this RISC's share of one block's sticks, not of the whole shard.
        const uint32_t input_num_pages_per_stick = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t input_num_sticks = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t input_write_offset = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t input_read_offset = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t input_block_stride = get_arg_val<uint32_t>(arg_idx++);

        DataflowBuffer input_dfb(input_id);
        const uint32_t input_base_read_addr = input_dfb.get_read_ptr() + input_read_offset;
        const uint32_t output_base_write_addr = base_l1_write_addr + input_write_offset;

        for (uint32_t block_idx = 0; block_idx < num_blocks; block_idx++) {
            uint32_t l1_write_addr = output_base_write_addr + block_idx * output_block_stride;
            uint32_t l1_read_addr = input_base_read_addr + block_idx * input_block_stride;

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
    }

    noc.async_read_barrier();
}
