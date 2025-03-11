// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // blank atm
    constexpr uint32_t shard_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t crop_w = get_compile_time_arg_val(1);
    constexpr uint32_t stick_size_in_bytes = get_compile_time_arg_val(2);

    constexpr uint32_t cropped_width_in_bytes = crop_w * stick_size_in_bytes;

    uint32_t src_addr = get_arg_val<uint32_t>(0);  // address of input tensor, pre cropping
    uint32_t write_offset = get_arg_val<uint32_t>(1);
    uint32_t num_reads = get_arg_val<uint32_t>(2);
    tt_l1_ptr uint32_t* args = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
    uint32_t args_idx = 0;

    // DPRINT <<  " write offset: " << write_offset << " num fully blown transfers: " << num_reads << "
    // stick_size_in_bytes: " << stick_size_in_bytes << " cropped_width_in_bytes:" << cropped_width_in_bytes << ENDL();

    uint32_t l1_write_addr = get_write_ptr(shard_cb_id) + write_offset;
    for (uint32_t i = 0; i < num_reads; ++i) {
        uint32_t bank_id = args[args_idx++];
        // DPRINT << "Initial row offset: " << args[args_idx] << ENDL();
        uint32_t addr = src_addr + args[args_idx++];
        uint32_t row_size_in_bytes = args[args_idx++];
        uint32_t num_rows_to_transfer = args[args_idx++];
        // DPRINT << "Transferring num_rows: " << num_rows_to_transfer << " row size in bytes: " << row_size_in_bytes <<
        // ENDL();
        for (uint32_t j = 0; j < num_rows_to_transfer; j++) {
            // Add crop width offset
            addr += cropped_width_in_bytes;
            noc_async_read(get_noc_addr_from_bank_id<false>(bank_id, addr), l1_write_addr, row_size_in_bytes);
            l1_write_addr += row_size_in_bytes;
            // Add crop width offset again, to crop last column
            addr += cropped_width_in_bytes + row_size_in_bytes;
        }
        // noc_async_read_barrier();
        //  //uint32_t read_size = args[args_idx++];
        //  DPRINT << "Bank id: " << bank_id << " addr: " << addr << " read size: " << read_size << ENDL();
        //  DPRINT << "Src addr: " << src_addr << " pre_incr_second_arg " << pre_incr_second_arg << ENDL();
        //  noc_async_read(get_noc_addr_from_bank_id<false>(bank_id, addr), l1_write_addr, read_size);
    }
    noc_async_read_barrier();
}
