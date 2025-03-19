// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t shard_cb_id = get_compile_time_arg_val(0);
    constexpr bool read_from_dram = get_compile_time_arg_val(1);

    const uint32_t total_num_sticks = get_arg_val<uint32_t>(0);
    const uint32_t local_stride_bytes = get_arg_val<uint32_t>(1);
    const uint32_t remote_stride_bytes = get_arg_val<uint32_t>(2);
    const uint32_t base_read_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_segments = get_arg_val<uint32_t>(4);

    uint32_t args_idx = 0;
    tt_l1_ptr uint32_t* args = (tt_l1_ptr uint32_t*)(get_arg_addr(5));

    uint32_t base_write_addr = get_read_ptr(shard_cb_id);

    for (uint32_t i = 0; i < num_segments; ++i) {
        uint32_t read_size = args[args_idx++];

        uint32_t write_offset = args[args_idx++];
        uint32_t l1_write_addr = base_write_addr + write_offset;

        uint32_t bank_id = args[args_idx++];
        uint32_t read_offset = base_read_addr + args[args_idx++];
        uint64_t noc_read_addr = get_noc_addr_from_bank_id<read_from_dram>(bank_id, read_offset);

        for (uint32_t j = 0; j < total_num_sticks; ++j) {
            noc_async_read(noc_read_addr, l1_write_addr, read_size);
            l1_write_addr += local_stride_bytes;
            noc_read_addr += remote_stride_bytes;
        }
    }
    noc_async_read_barrier();
}
