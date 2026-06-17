// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of the height-only sharded pad reader (private to PadRmShardedHeightOnlyProgramFactory).
// Device-side NoC logic is unchanged; resource access moves to the Metal 2.0 named handles
// (dfb::/args::):
//   - c_0 input shard  -> dfb::cb_in0  (borrowed-from-input fake CB: used only for its base address to
//                         issue remote NoC reads against neighbouring shards; bound as a self-loop).
//   - c_16 output shard -> dfb::cb_out0 (borrowed-from-output; reader PRODUCER, writer CONSUMER).
//   - The legacy variable-length arg tail (read_noc_x/y, num_stick_chunks, chunk lists) is read with
//     runtime-computed indices, so it becomes runtime varargs (padded to a uniform max by the host).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t stick_size_bytes = get_arg(args::stick_size_bytes);
    constexpr uint32_t num_sticks_padded = get_arg(args::num_sticks_padded);

    const uint32_t num_cores_read = get_arg(args::num_cores_read);

    // Vararg tail layout (per core; padded to a uniform max):
    //   [ read_noc_x/y interleaved : 2*num_cores_read ]
    //   [ num_stick_chunks         :   num_cores_read ]
    //   [ (chunk_start_id, chunk_num_sticks) pairs : 2 * sum(num_stick_chunks) ]
    const uint32_t num_stick_chunks_base = num_cores_read * 2;
    const uint32_t chunk_base = num_cores_read * 3;

    DataflowBuffer cb_in0(dfb::cb_in0);
    DataflowBuffer cb_out0(dfb::cb_out0);
    Noc noc;

    cb_out0.reserve_back(num_sticks_padded);
    uint32_t l1_read_addr = cb_in0.get_write_ptr();
    uint32_t l1_write_addr = cb_out0.get_write_ptr();

    uint32_t chunk_ptr_offset = 0;
    uint32_t read_noc_xy_ptr_offset = 0;

    for (uint32_t curr_core = 0; curr_core < num_cores_read; ++curr_core) {
        const uint32_t src_noc_x = get_vararg(read_noc_xy_ptr_offset);
        const uint32_t src_noc_y = get_vararg(read_noc_xy_ptr_offset + 1);

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
    cb_out0.push_back(num_sticks_padded);
}
