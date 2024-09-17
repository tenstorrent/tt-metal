// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
	constexpr uint32_t shard_cb_id = get_compile_time_arg_val(0);

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t read_offset = get_arg_val<uint32_t>(1);
    uint32_t num_writes = get_arg_val<uint32_t>(2);
    tt_l1_ptr uint32_t * args = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
    uint32_t args_idx = 0;

    uint32_t l1_read_addr = get_read_ptr(shard_cb_id) + read_offset;
    for (uint32_t i = 0; i < num_writes; ++i) {
        uint32_t x_coord = args[args_idx++];
        uint32_t y_coord = args[args_idx++];
        uint32_t addr = dst_addr + args[args_idx++];
        uint64_t dst_noc_addr = get_noc_addr(x_coord, y_coord, addr);
        uint32_t write_size = args[args_idx++];
        noc_async_write(l1_read_addr, dst_noc_addr, write_size);
        l1_read_addr += write_size;
    }
    noc_async_write_barrier();
}
