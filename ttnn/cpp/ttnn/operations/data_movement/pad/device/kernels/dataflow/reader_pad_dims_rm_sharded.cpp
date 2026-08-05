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
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t num_sticks_padded = get_compile_time_arg_val(1);

    const uint32_t num_cores_read = get_arg_val<uint32_t>(0);
    tt_l1_ptr uint32_t* read_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(1));
    tt_l1_ptr uint32_t* read_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
    tt_l1_ptr uint32_t* num_stick_chunks = (tt_l1_ptr uint32_t*)(get_arg_addr(1 + num_cores_read * 2));
    tt_l1_ptr uint32_t* chunk_start_id = (tt_l1_ptr uint32_t*)(get_arg_addr(1 + num_cores_read * 3));
    tt_l1_ptr uint32_t* chunk_num_sticks = (tt_l1_ptr uint32_t*)(chunk_start_id + 1);

    constexpr auto dfb_in0 = tt::CBIndex::c_0;
    constexpr auto dfb_out0 = tt::CBIndex::c_16;
    DataflowBuffer dfb_in0_exp(dfb_in0);
    DataflowBuffer dfb_out0_exp(dfb_out0);

    Noc noc;

    dfb_out0_exp.reserve_back(num_sticks_padded);
    uint32_t l1_read_addr = dfb_in0_exp.get_write_ptr();
    uint32_t l1_write_addr = dfb_out0_exp.get_write_ptr();

    uint32_t chunk_ptr_offset = 0;
    uint32_t read_noc_xy_ptr_offset = 0;

    for (uint32_t curr_core = 0; curr_core < num_cores_read; ++curr_core) {
        const uint32_t src_noc_x = read_noc_x[read_noc_xy_ptr_offset];
        const uint32_t src_noc_y = read_noc_y[read_noc_xy_ptr_offset];

        uint32_t curr_core_num_chunks = num_stick_chunks[curr_core];

        for (uint32_t curr_chunk = 0; curr_chunk < curr_core_num_chunks; ++curr_chunk) {
            uint32_t curr_start_id = chunk_start_id[chunk_ptr_offset];
            uint32_t curr_num_sticks = chunk_num_sticks[chunk_ptr_offset];

            uint32_t l1_read_offset = curr_start_id * stick_size_bytes;
            uint32_t read_data_size_bytes = curr_num_sticks * stick_size_bytes;

            if ((curr_start_id != (uint32_t)-1) and (curr_start_id != (uint32_t)-2)) {
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(
                    UnicastEndpoint{},
                    dst,
                    read_data_size_bytes,
                    {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = l1_read_addr + l1_read_offset},
                    {.offset_bytes = 0});
            }

            l1_write_addr += read_data_size_bytes;
            chunk_ptr_offset += 2;
        }

        read_noc_xy_ptr_offset += 2;
    }

    noc.async_read_barrier();
    dfb_out0_exp.push_back(num_sticks_padded);
}
