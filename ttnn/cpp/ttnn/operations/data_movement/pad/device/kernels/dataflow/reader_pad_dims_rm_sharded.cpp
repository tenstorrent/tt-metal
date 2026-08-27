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
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto stick_size_bytes = get_arg(args::stick_size_bytes);
    constexpr auto num_sticks_padded = get_arg(args::num_sticks_padded);

    // The gather plan is data-directed, so it arrives as a runtime vararg block laid out as:
    //   [0, 2*num_cores_read)              interleaved NoC (x, y) per source core
    //   [2*num_cores_read, 3*num_cores_read)  chunk count per source core
    //   [3*num_cores_read, ...)            a (start_id, length) pair per chunk
    const auto num_cores_read = get_arg(args::num_cores_read);
    const uint32_t noc_xy_base = 0;
    const uint32_t num_stick_chunks_base = num_cores_read * 2;
    const uint32_t chunk_base = num_cores_read * 3;

    DataflowBuffer dfb_in0_exp(dfb::in_shard);
    DataflowBuffer dfb_out0_exp(dfb::out_shard);

    Noc noc;

    dfb_out0_exp.reserve_back(num_sticks_padded);
    uint32_t l1_read_addr = dfb_in0_exp.get_write_ptr();
    uint32_t l1_write_addr = dfb_out0_exp.get_write_ptr();

    uint32_t chunk_ptr_offset = 0;
    uint32_t read_noc_xy_ptr_offset = 0;

    for (uint32_t curr_core = 0; curr_core < num_cores_read; ++curr_core) {
        const uint32_t src_noc_x = get_vararg(noc_xy_base + read_noc_xy_ptr_offset);
        const uint32_t src_noc_y = get_vararg(noc_xy_base + read_noc_xy_ptr_offset + 1);

        uint32_t curr_core_num_chunks = get_vararg(num_stick_chunks_base + curr_core);

        for (uint32_t curr_chunk = 0; curr_chunk < curr_core_num_chunks; ++curr_chunk) {
            uint32_t curr_start_id = get_vararg(chunk_base + chunk_ptr_offset);
            uint32_t curr_num_sticks = get_vararg(chunk_base + chunk_ptr_offset + 1);

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
