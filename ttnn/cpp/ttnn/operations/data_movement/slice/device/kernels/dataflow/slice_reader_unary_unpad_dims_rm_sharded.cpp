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
    constexpr uint32_t stick_size_unpadded = get_compile_time_arg_val(0);
    constexpr uint32_t num_sticks_unpadded = get_compile_time_arg_val(1);
    // Buffer's aligned row stride (>= payload when W·E is not 16-aligned); begins_bytes = slice_start[-1] * E.
    constexpr uint32_t src_stride_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t dst_stride_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t begins_bytes = get_compile_time_arg_val(4);

    // One coalesced read only when every row is contiguous on both sides and starts at column 0.
    constexpr bool can_coalesce =
        (begins_bytes == 0) && (src_stride_bytes == stick_size_unpadded) && (dst_stride_bytes == stick_size_unpadded);

    const uint32_t num_cores_read = get_arg_val<uint32_t>(0);
    tt_l1_ptr uint32_t* read_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(1));
    tt_l1_ptr uint32_t* read_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
    tt_l1_ptr uint32_t* num_stick_chunks = (tt_l1_ptr uint32_t*)(get_arg_addr(1 + num_cores_read * 2));
    tt_l1_ptr uint32_t* chunk_start_id = (tt_l1_ptr uint32_t*)(get_arg_addr(1 + num_cores_read * 3));
    tt_l1_ptr uint32_t* chunk_num_sticks = (tt_l1_ptr uint32_t*)(chunk_start_id + 1);

    constexpr auto dfb_in0 = tt::CBIndex::c_0;
    constexpr auto dfb_out0 = tt::CBIndex::c_16;

    Noc noc;
    // Create DataflowBuffers for Device 2.0 API
    DataflowBuffer dfb_in(dfb_in0);
    DataflowBuffer dfb_out(dfb_out0);

    dfb_out.reserve_back(num_sticks_unpadded);
    uint32_t l1_read_addr = dfb_in.get_write_ptr();
    uint32_t l1_write_addr = dfb_out.get_write_ptr();

    uint32_t chunk_ptr_offset = 0;
    uint32_t read_noc_xy_ptr_offset = 0;

    for (uint32_t curr_core = 0; curr_core < num_cores_read; ++curr_core) {
        const uint32_t src_noc_x = read_noc_x[read_noc_xy_ptr_offset];
        const uint32_t src_noc_y = read_noc_y[read_noc_xy_ptr_offset];

        uint32_t curr_core_num_chunks = num_stick_chunks[curr_core];

        for (uint32_t curr_chunk = 0; curr_chunk < curr_core_num_chunks; ++curr_chunk) {
            uint32_t curr_start_id = chunk_start_id[chunk_ptr_offset];
            uint32_t curr_num_sticks = chunk_num_sticks[chunk_ptr_offset];

            if constexpr (can_coalesce) {
                uint32_t src_off = curr_start_id * src_stride_bytes;
                uint32_t bytes = curr_num_sticks * stick_size_unpadded;
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(
                    UnicastEndpoint{},
                    dst,
                    bytes,
                    {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = l1_read_addr + src_off},
                    {.offset_bytes = 0});
                l1_write_addr += curr_num_sticks * dst_stride_bytes;
            } else {
                uint32_t src_off = curr_start_id * src_stride_bytes + begins_bytes;
                for (uint32_t s = 0; s < curr_num_sticks; ++s) {
                    CoreLocalMem<uint32_t> dst(l1_write_addr);
                    noc.async_read(
                        UnicastEndpoint{},
                        dst,
                        stick_size_unpadded,
                        {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = l1_read_addr + src_off},
                        {.offset_bytes = 0});
                    src_off += src_stride_bytes;
                    l1_write_addr += dst_stride_bytes;
                }
            }
            chunk_ptr_offset += 2;
        }

        read_noc_xy_ptr_offset += 2;
    }

    noc.async_read_barrier();
    dfb_out.push_back(num_sticks_unpadded);
}
