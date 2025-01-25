// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t shard_cb_id = get_compile_time_arg_val(0);
    constexpr bool read_from_dram = get_compile_time_arg_val(1);
    constexpr uint32_t scratch_cb_id = get_compile_time_arg_val(2);

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t write_offset = get_arg_val<uint32_t>(1);
    uint32_t num_reads = get_arg_val<uint32_t>(2);
    tt_l1_ptr uint32_t* args = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
    uint32_t args_idx = 0;

    uint32_t l1_write_addr = get_write_ptr(shard_cb_id) + write_offset;
    cb_reserve_back(scratch_cb_id, 1);
    uint32_t scratch_l1_write_addr = get_write_ptr(scratch_cb_id);
    uint64_t scratch_l1_noc_read_addr = get_noc_addr(scratch_l1_write_addr);

    for (uint32_t i = 0; i < num_reads; ++i) {
        uint32_t bank_id = args[args_idx++];
        uint32_t addr = src_addr + args[args_idx++];
        uint32_t units_to_transfer = args[args_idx++];
        uint32_t unit_size = args[args_idx++];
        uint32_t read_stride_bytes = args[args_idx++];
        uint32_t write_stride_bytes = args[args_idx++];
        uint64_t read_addr = get_noc_addr_from_bank_id<read_from_dram>(bank_id, addr);
        for (uint32_t unit_idx = 0; unit_idx < units_to_transfer; ++unit_idx) {
            noc_async_read(read_addr, scratch_l1_write_addr, unit_size);
            noc_async_read_barrier();
            noc_async_read(scratch_l1_noc_read_addr, l1_write_addr, unit_size);
            read_addr += read_stride_bytes;
            l1_write_addr += write_stride_bytes;
            noc_async_read_barrier();
        }
    }
}
