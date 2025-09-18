// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

	constexpr uint32_t shard_cb_id = get_compile_time_arg_val(0);
    constexpr bool write_to_dram = get_compile_time_arg_val(1);
    constexpr bool unaligned = get_compile_time_arg_val(2);
    constexpr uint32_t unit_size = get_compile_time_arg_val(3);
    constexpr uint32_t local_unit_size_padded = get_compile_time_arg_val(4);
    constexpr uint32_t remote_unit_size_padded = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scratch_index = get_compile_time_arg_val(6);

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t read_offset = get_arg_val<uint32_t>(1);
    uint32_t num_writes = get_arg_val<uint32_t>(2);
    tt_l1_ptr uint32_t* args = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
    uint32_t args_idx = 0;

    uint32_t l1_read_addr = get_read_ptr(shard_cb_id) + read_offset;
    for (uint32_t i = 0; i < num_writes; ++i) {
        uint32_t bank_id = args[args_idx++];
        uint32_t addr = dst_addr + args[args_idx++];
        uint32_t units_to_transfer = args[args_idx++];
        uint32_t write_size = units_to_transfer * unit_size;
        noc_async_write(l1_read_addr, get_noc_addr_from_bank_id<write_to_dram>(bank_id, addr), write_size);
        l1_read_addr += write_size;
    }
    noc_async_write_barrier();
}
