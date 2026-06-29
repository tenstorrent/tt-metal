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
    constexpr uint32_t stick_size_padded = get_arg(args::stick_size_padded);
    constexpr uint32_t stick_size_unpadded = get_arg(args::stick_size_unpadded);
    constexpr uint32_t num_sticks_unpadded = get_arg(args::num_sticks_unpadded);

    // The reader's per-core runtime arguments are a single variable-length blob, supplied
    // positionally as runtime varargs (the length varies per core: num_cores_read, the chunk
    // counts, and the chunk lists all differ). The vararg layout mirrors the legacy pointer
    // walking exactly:
    //   vararg[0]                              = num_cores_read
    //   vararg[1 .. 1 + 2*num_cores_read)      = (noc_x, noc_y) pairs, stride 2
    //   vararg[1 + 2*num_cores_read .. 1 + 3*num_cores_read) = num_stick_chunks[core]
    //   vararg[1 + 3*num_cores_read ...)       = (chunk_start_id, chunk_num_sticks) pairs
    const uint32_t num_cores_read = get_vararg(0);
    const uint32_t noc_xy_base = 1;
    const uint32_t num_chunks_base = 1 + num_cores_read * 2;
    const uint32_t chunk_base = 1 + num_cores_read * 3;

    Noc noc;
    // Create DataflowBuffers for Device 2.0 API.
    // cb_in is borrowed onto the input shard buffer (resident in L1); cb_out is borrowed
    // onto the output shard buffer. The reader reads from cb_in's L1 base via raw noc x/y
    // addresses and writes the unpadded sticks into cb_out.
    DataflowBuffer cb_in(dfb::cb_in);
    DataflowBuffer cb_out(dfb::cb_out);

    cb_out.reserve_back(num_sticks_unpadded);
    uint32_t l1_read_addr = cb_in.get_write_ptr();
    uint32_t l1_write_addr = cb_out.get_write_ptr();

    uint32_t chunk_ptr_offset = 0;  // running chunk index (in chunk pairs) across all cores

    for (uint32_t curr_core = 0; curr_core < num_cores_read; ++curr_core) {
        const uint32_t src_noc_x = get_vararg(noc_xy_base + curr_core * 2);
        const uint32_t src_noc_y = get_vararg(noc_xy_base + curr_core * 2 + 1);

        uint32_t curr_core_num_chunks = get_vararg(num_chunks_base + curr_core);

        for (uint32_t curr_chunk = 0; curr_chunk < curr_core_num_chunks; ++curr_chunk) {
            uint32_t curr_start_id = get_vararg(chunk_base + chunk_ptr_offset * 2);
            uint32_t curr_num_sticks = get_vararg(chunk_base + chunk_ptr_offset * 2 + 1);

            uint32_t l1_read_offset = curr_start_id * stick_size_unpadded;
            uint32_t read_data_size_bytes = curr_num_sticks * stick_size_unpadded;

            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(
                UnicastEndpoint{},
                dst,
                read_data_size_bytes,
                {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = l1_read_addr + l1_read_offset},
                {.offset_bytes = 0});
            l1_write_addr += read_data_size_bytes;
            chunk_ptr_offset += 1;
        }
    }

    noc.async_read_barrier();
    cb_out.push_back(num_sticks_unpadded);
}
