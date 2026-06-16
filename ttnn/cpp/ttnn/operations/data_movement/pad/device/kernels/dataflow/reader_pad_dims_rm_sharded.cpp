// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port (in place — used only by the PadRmShardedHeightOnly factory).
// Logic UNCHANGED; only the access mechanism moves to named bindings:
//   - input shard CB c_0  -> dfb::in0  (borrowed input shard; read by base pointer only -> self-loop)
//   - output shard CB c_16 -> dfb::out0 (borrowed output shard; reader produces, writer consumes)
//   - positional CTAs      -> get_arg(args::...)
//   - the packed variable-length RTA tail (per-core noc x/y + stick chunks) -> runtime varargs.
// The legacy reader read this tail via get_arg_addr pointer arithmetic; the same packed layout is
// preserved, now sourced from varargs (num_cores_read is a named RTA, the rest is the vararg tail).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t stick_size_bytes = get_arg(args::stick_size_bytes);
    constexpr uint32_t num_sticks_padded = get_arg(args::num_sticks_padded);

    const uint32_t num_cores_read = get_arg(args::num_cores_read);
    // Packed vararg tail, preserving the legacy get_arg_addr layout:
    //   [0 .. 2*num_cores)        : read_noc_x / read_noc_y interleaved (stride 2)
    //   [2*num_cores .. 3*num_cores) : num_stick_chunks (one per core)
    //   [3*num_cores .. )         : chunk_start_id / chunk_num_sticks interleaved (stride 2)
    const uint32_t noc_xy_base = 0;
    const uint32_t num_chunks_base = num_cores_read * 2;
    const uint32_t chunks_base = num_cores_read * 3;

    constexpr auto cb_in0 = dfb::in0;
    constexpr auto cb_out0 = dfb::out0;
    DataflowBuffer cb_in0_exp(cb_in0);
    DataflowBuffer cb_out0_exp(cb_out0);

    Noc noc;

    cb_out0_exp.reserve_back(num_sticks_padded);
    uint32_t l1_read_addr = cb_in0_exp.get_write_ptr();
    uint32_t l1_write_addr = cb_out0_exp.get_write_ptr();

    uint32_t chunk_ptr_offset = 0;
    uint32_t read_noc_xy_ptr_offset = 0;

    for (uint32_t curr_core = 0; curr_core < num_cores_read; ++curr_core) {
        const uint32_t src_noc_x = get_vararg(noc_xy_base + read_noc_xy_ptr_offset);
        const uint32_t src_noc_y = get_vararg(noc_xy_base + read_noc_xy_ptr_offset + 1);

        uint32_t curr_core_num_chunks = get_vararg(num_chunks_base + curr_core);

        for (uint32_t curr_chunk = 0; curr_chunk < curr_core_num_chunks; ++curr_chunk) {
            uint32_t curr_start_id = get_vararg(chunks_base + chunk_ptr_offset);
            uint32_t curr_num_sticks = get_vararg(chunks_base + chunk_ptr_offset + 1);

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
    cb_out0_exp.push_back(num_sticks_padded);
}
