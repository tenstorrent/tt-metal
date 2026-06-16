// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 (ProgramSpec) port of slice_reader_unary_unpad_dims_rm_sharded.cpp.
// Used only by SliceRmShardedSpecProgramFactory, so it is a local fork of the legacy reader
// (the legacy file is still consumed unchanged by SliceRmShardedProgramFactory::create_descriptor,
// which ccl/mesh_partition reuses). Logic, loop bounds and numeric paths are UNCHANGED; only the
// access mechanism moves to named bindings.
//
// This is a Case-2 (bridge) port: the kernel performs a hand-rolled NoC walk over a
// host-computed physical core-coordinate map (read_noc_x/read_noc_y) and reads peer shards by
// raw {noc_x, noc_y, addr}. The input/output shard L1 base addresses are obtained from the two
// borrowed-memory DFBs (cb_in / cb_out, .borrowed_from src / dst), exactly as the legacy kernel
// obtained them from the borrowed CBs via get_write_ptr(). The raw NoC arithmetic is kept
// verbatim; only the resource construction and arg retrieval change:
//   CB ids               -> dfb::cb_in / dfb::cb_out (borrowed onto the src/dst tensors)
//   stick_size CTAs      -> named CTAs (get_arg(args::...))
//   per-core NoC map / chunk descriptors -> runtime varargs (get_vararg), identical layout
//
// Vararg note: the legacy kernel indexed the per-core RTA region directly through tt_l1_ptr
// pointers computed from num_cores_read. The m2 vararg API exposes value getters only, so the
// same indices are read with get_vararg(i); the arithmetic that derives each offset is unchanged.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t stick_size_padded = get_arg(args::stick_size_padded);
    constexpr uint32_t stick_size_unpadded = get_arg(args::stick_size_unpadded);
    constexpr uint32_t num_sticks_unpadded = get_arg(args::num_sticks_unpadded);

    // Per-core runtime varargs layout (identical to the legacy RTA layout):
    //   [0]                          : num_cores_read
    //   [1 + i*2 .. ], i in [0,num_cores_read) : read_noc_x[i], read_noc_y[i]
    //   [1 + num_cores_read*2 + i]   : num_stick_chunks[i]
    //   [1 + num_cores_read*3 ..]    : (chunk_start_id, chunk_num_sticks) pairs
    const uint32_t num_cores_read = get_vararg(0);
    const uint32_t noc_xy_base = 1;
    const uint32_t num_chunks_base = 1 + num_cores_read * 2;
    const uint32_t chunk_base = 1 + num_cores_read * 3;

    constexpr uint32_t cb_id_in0 = dfb::cb_in;
    constexpr uint32_t cb_id_out0 = dfb::cb_out;

    Noc noc;
    // Create CircularBuffers for Device 2.0 API
    CircularBuffer cb_in(cb_id_in0);
    CircularBuffer cb_out(cb_id_out0);

    cb_out.reserve_back(num_sticks_unpadded);
    uint32_t l1_read_addr = cb_in.get_write_ptr();
    uint32_t l1_write_addr = cb_out.get_write_ptr();

    uint32_t chunk_ptr_offset = 0;
    uint32_t read_noc_xy_ptr_offset = 0;

    for (uint32_t curr_core = 0; curr_core < num_cores_read; ++curr_core) {
        const uint32_t src_noc_x = get_vararg(noc_xy_base + read_noc_xy_ptr_offset);
        const uint32_t src_noc_y = get_vararg(noc_xy_base + read_noc_xy_ptr_offset + 1);

        uint32_t curr_core_num_chunks = get_vararg(num_chunks_base + curr_core);

        for (uint32_t curr_chunk = 0; curr_chunk < curr_core_num_chunks; ++curr_chunk) {
            uint32_t curr_start_id = get_vararg(chunk_base + chunk_ptr_offset);
            uint32_t curr_num_sticks = get_vararg(chunk_base + chunk_ptr_offset + 1);

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
            chunk_ptr_offset += 2;
        }

        read_noc_xy_ptr_offset += 2;
    }

    noc.async_read_barrier();
    cb_out.push_back(num_sticks_unpadded);
}
