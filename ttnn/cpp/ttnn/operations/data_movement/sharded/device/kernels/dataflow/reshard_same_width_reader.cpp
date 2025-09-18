// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint_pages.h"

void kernel_main() {
    constexpr uint32_t shard_cb_id = get_compile_time_arg_val(0);
    constexpr bool read_from_dram = get_compile_time_arg_val(1);
    constexpr bool unaligned = get_compile_time_arg_val(2);
    constexpr uint32_t unit_size = get_compile_time_arg_val(3);
    constexpr uint32_t local_unit_size_padded = get_compile_time_arg_val(4);
    constexpr uint32_t remote_unit_size_padded = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scratch_index = get_compile_time_arg_val(6);

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t write_offset = get_arg_val<uint32_t>(1);
    uint32_t num_reads = get_arg_val<uint32_t>(2);
    tt_l1_ptr uint32_t* args = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
    uint32_t args_idx = 0;

    uint32_t l1_write_addr = get_write_ptr(shard_cb_id) + write_offset;
    if constexpr (unaligned) {
        uint32_t l1_scratch_write_addr = get_write_ptr(cb_scratch_index);
        uint32_t l1_scratch_read_addr = get_read_ptr(cb_scratch_index);
        for (uint32_t i = 0; i < num_reads; ++i) {
            uint32_t bank_id = args[args_idx++];
            uint32_t src_offset = args[args_idx++];
            uint32_t addr = src_addr + src_offset;
            DPRINT << "addr: " << addr << ENDL();
            uint32_t units_to_transfer = args[args_idx++];
            uint32_t read_size = units_to_transfer * remote_unit_size_padded;
            noc_async_read(
                get_noc_addr_from_bank_id<read_from_dram>(bank_id, addr),
                l1_scratch_write_addr + src_offset,
                read_size);
            noc_async_read_barrier();
            // tt::data_movement::common::print_bf16_pages(
            //     l1_scratch_write_addr + src_offset, remote_unit_size_padded / 2, units_to_transfer);

            // uint64_t pad_align_noc_addr = get_noc_addr(l1_scratch_read_addr + j * remote_unit_size_padded);
            uint64_t pad_align_noc_addr = get_noc_addr(l1_scratch_read_addr + src_offset);
            for (uint32_t j = 0; j < units_to_transfer; ++j) {
                noc_async_read(pad_align_noc_addr, l1_write_addr, unit_size);
                // tt::data_movement::common::print_bf16_pages(l1_write_addr, unit_size / 2, 1);
                l1_write_addr += unit_size;
                pad_align_noc_addr += remote_unit_size_padded;
            }
            noc_async_read_barrier();
        }
    } else {
        for (uint32_t i = 0; i < num_reads; ++i) {
            uint32_t bank_id = args[args_idx++];
            uint32_t addr = src_addr + args[args_idx++];
            uint32_t units_to_transfer = args[args_idx++];
            uint32_t read_size = units_to_transfer * unit_size;
            noc_async_read(get_noc_addr_from_bank_id<read_from_dram>(bank_id, addr), l1_write_addr, read_size);
            l1_write_addr += read_size;
        }
        noc_async_read_barrier();
    }
}
