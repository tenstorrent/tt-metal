// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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
    constexpr uint32_t shard_dfb_id = get_compile_time_arg_val(0);
    constexpr bool write_to_dram = get_compile_time_arg_val(1);
    constexpr AllocatorBankType bank_type = write_to_dram ? AllocatorBankType::DRAM : AllocatorBankType::L1;

    const uint32_t total_num_sticks = get_arg_val<uint32_t>(0);
    const uint32_t local_stride_bytes = get_arg_val<uint32_t>(1);
    const uint32_t remote_stride_bytes = get_arg_val<uint32_t>(2);
    const uint32_t base_write_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_segments = get_arg_val<uint32_t>(4);

    uint32_t args_idx = 0;
    tt_l1_ptr uint32_t* args = (tt_l1_ptr uint32_t*)(get_arg_addr(5));

    DataflowBuffer shard_dfb(shard_dfb_id);
    Noc noc;
    AllocatorBank<bank_type> bank;

    uint32_t base_l1_read_addr = shard_dfb.get_read_ptr();

    for (uint32_t i = 0; i < num_segments; ++i) {
        uint32_t write_size = args[args_idx++];

        uint32_t read_offset = args[args_idx++];
        uint32_t l1_read_addr = base_l1_read_addr + read_offset;

        uint32_t bank_id = args[args_idx++];
        uint32_t write_offset = base_write_addr + args[args_idx++];

        for (uint32_t j = 0; j < total_num_sticks; ++j) {
            CoreLocalMem<uint32_t> src(l1_read_addr);
            noc.async_write(src, bank, write_size, {.offset_bytes = 0}, {.bank_id = bank_id, .addr = write_offset});
            l1_read_addr += local_stride_bytes;
            write_offset += remote_stride_bytes;
        }
    }
    noc.async_write_barrier();
}
