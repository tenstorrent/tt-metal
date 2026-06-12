// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 sharded row-major slice reader. Device-side dataflow — the Noc object, the
// CoreLocalMem shard-to-shard reads, and the two CircularBuffers — is preserved verbatim from
// the descriptor-era slice_reader_unary_unpad_dims_rm_sharded.cpp; only the binding mechanisms
// change:
//   - CB indices            -> dfb::cb_in / dfb::cb_out (both BORROWED from the sharded
//                              input/output tensors; the borrowed-DFB machinery sets each CB's
//                              backing L1 address from its tensor argument)
//   - stick sizes / height  -> named CTAs (get_named_compile_time_arg_val), in the cache key
//   - per-core arg blob      -> a single VARIABLE-LENGTH runtime vararg vector, decoded
//                              positionally (get_vararg), exactly as the legacy kernel decoded
//                              its positional RTA region. Layout (per core):
//                                [0]                         num_cores_read (N)
//                                [1 .. 1+2N)                 N (noc_x, noc_y) pairs
//                                [1+2N .. 1+3N)              N per-core chunk counts
//                                [1+3N .. )                  (chunk_start, chunk_len) pairs
//
// There is no TensorAccessor and no tensor_binding accessor: the shard-to-shard reads use
// explicit (noc_x, noc_y, l1_addr) endpoints built from the borrowed CB write pointers.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t stick_size_padded = get_named_compile_time_arg_val("stick_size_padded");
    constexpr uint32_t stick_size_unpadded = get_named_compile_time_arg_val("stick_size_unpadded");
    constexpr uint32_t num_sticks_unpadded = get_named_compile_time_arg_val("num_sticks_unpadded");

    const uint32_t num_cores_read = get_vararg(0);
    // noc coords: pair k at varargs [1 + 2k, 2 + 2k]
    const uint32_t noc_section_base = 1;
    // chunk counts: one per core at varargs [1 + 2N + k]
    const uint32_t chunk_count_base = 1 + num_cores_read * 2;
    // (chunk_start, chunk_len) pairs at varargs [1 + 3N + 2c, 1 + 3N + 2c + 1]
    const uint32_t chunk_data_base = 1 + num_cores_read * 3;

    constexpr uint32_t cb_in0 = dfb::cb_in;
    constexpr uint32_t cb_out0 = dfb::cb_out;

    Noc noc;
    CircularBuffer cb_in(cb_in0);
    CircularBuffer cb_out(cb_out0);

    cb_out.reserve_back(num_sticks_unpadded);
    uint32_t l1_read_addr = cb_in.get_write_ptr();
    uint32_t l1_write_addr = cb_out.get_write_ptr();

    uint32_t chunk_data_offset = chunk_data_base;

    for (uint32_t curr_core = 0; curr_core < num_cores_read; ++curr_core) {
        const uint32_t src_noc_x = get_vararg(noc_section_base + curr_core * 2);
        const uint32_t src_noc_y = get_vararg(noc_section_base + curr_core * 2 + 1);

        const uint32_t curr_core_num_chunks = get_vararg(chunk_count_base + curr_core);

        for (uint32_t curr_chunk = 0; curr_chunk < curr_core_num_chunks; ++curr_chunk) {
            const uint32_t curr_start_id = get_vararg(chunk_data_offset);
            const uint32_t curr_num_sticks = get_vararg(chunk_data_offset + 1);

            const uint32_t l1_read_offset = curr_start_id * stick_size_unpadded;
            const uint32_t read_data_size_bytes = curr_num_sticks * stick_size_unpadded;

            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(
                UnicastEndpoint{},
                dst,
                read_data_size_bytes,
                {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = l1_read_addr + l1_read_offset},
                {.offset_bytes = 0});
            l1_write_addr += read_data_size_bytes;
            chunk_data_offset += 2;
        }
    }

    noc.async_read_barrier();
    cb_out.push_back(num_sticks_unpadded);
}
