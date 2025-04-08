// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t stick_size_padded = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size_unpadded = get_compile_time_arg_val(1);
    constexpr uint32_t num_sticks_unpadded = get_compile_time_arg_val(2);

    const uint32_t num_cores_read = get_arg_val<uint32_t>(0);
    tt_l1_ptr uint32_t* read_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(1));
    tt_l1_ptr uint32_t* read_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
    tt_l1_ptr uint32_t* num_stick_chunks = (tt_l1_ptr uint32_t*)(get_arg_addr(1 + num_cores_read * 2));
    tt_l1_ptr uint32_t* chunk_start_id = (tt_l1_ptr uint32_t*)(get_arg_addr(1 + num_cores_read * 3));
    tt_l1_ptr uint32_t* chunk_num_sticks = (tt_l1_ptr uint32_t*)(chunk_start_id + 1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    cb_reserve_back(cb_out0, num_sticks_unpadded);
    uint32_t l1_read_addr = get_write_ptr(cb_in0);
    uint32_t l1_write_addr = get_write_ptr(cb_out0);

    uint32_t chunk_ptr_offset = 0;
    uint32_t read_noc_xy_ptr_offset = 0;

    for (uint32_t curr_core = 0; curr_core < num_cores_read; ++curr_core) {
        uint64_t src_noc_addr =
            get_noc_addr(read_noc_x[read_noc_xy_ptr_offset], read_noc_y[read_noc_xy_ptr_offset], l1_read_addr);

        uint32_t curr_core_num_chunks = num_stick_chunks[curr_core];

        for (uint32_t curr_chunk = 0; curr_chunk < curr_core_num_chunks; ++curr_chunk) {
            uint32_t curr_start_id = chunk_start_id[chunk_ptr_offset];
            uint32_t curr_num_sticks = chunk_num_sticks[chunk_ptr_offset];

            uint32_t l1_read_offset = curr_start_id * stick_size_unpadded;
            uint32_t read_data_size_bytes = curr_num_sticks * stick_size_unpadded;

            noc_async_read(src_noc_addr + l1_read_offset, l1_write_addr, read_data_size_bytes);
            l1_write_addr += read_data_size_bytes;
            chunk_ptr_offset += 2;
        }

        read_noc_xy_ptr_offset += 2;
    }

    noc_async_read_barrier();
    cb_push_back(cb_out0, num_sticks_unpadded);
}
