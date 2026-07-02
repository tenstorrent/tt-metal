// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0: the resident input/output shards are reached by L1 base address from local
// TensorAccessors (tensor::input / tensor::output) — no borrowed self-loop CBs — plus named CTAs.
// The per-core packed work description (source core NOC coords and per-source stick chunks) is a
// runtime vararg, read positionally via a running index.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t stick_size_padded = get_arg(args::stick_size_padded);
    constexpr uint32_t stick_size_unpadded = get_arg(args::stick_size_unpadded);

    // Per-core packed work description (runtime vararg), positional layout:
    //   [0]                              num_cores_read
    //   [1 .. 1 + 2*num_cores_read)      (noc_x, noc_y) pairs, one per source core
    //   [.. + num_cores_read)            num_stick_chunks, one per source core
    //   [.. ]                            (chunk_start_id, chunk_num_sticks) pairs, per chunk
    uint32_t vararg_idx = 0;
    const uint32_t num_cores_read = get_vararg(vararg_idx++);

    Noc noc;
    // Resident input/output shards, read/written by L1 base address from local TensorAccessors
    // (no borrowed self-loop CBs).
    const auto s_in = TensorAccessor(tensor::input);
    const auto s_out = TensorAccessor(tensor::output);
    uint32_t l1_read_addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(s_in.get_noc_addr(0));
    uint32_t l1_write_addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(s_out.get_noc_addr(0));

    // The (noc_x, noc_y) pairs occupy [1 .. 1 + 2*num_cores_read); the num_stick_chunks block
    // begins right after, and the (start_id, num_sticks) chunk pairs after that. We read the noc
    // coords and chunk counts up front, then stream the chunk pairs in order.
    const uint32_t noc_coords_base = vararg_idx;                            // index of first noc_x
    const uint32_t num_chunks_base = noc_coords_base + 2 * num_cores_read;  // index of first num_stick_chunks
    uint32_t chunk_pair_idx = num_chunks_base + num_cores_read;             // index of first chunk (start,len) pair

    for (uint32_t curr_core = 0; curr_core < num_cores_read; ++curr_core) {
        const uint32_t src_noc_x = get_vararg(noc_coords_base + 2 * curr_core);
        const uint32_t src_noc_y = get_vararg(noc_coords_base + 2 * curr_core + 1);

        const uint32_t curr_core_num_chunks = get_vararg(num_chunks_base + curr_core);

        for (uint32_t curr_chunk = 0; curr_chunk < curr_core_num_chunks; ++curr_chunk) {
            const uint32_t curr_start_id = get_vararg(chunk_pair_idx);
            const uint32_t curr_num_sticks = get_vararg(chunk_pair_idx + 1);
            chunk_pair_idx += 2;

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
        }
    }

    noc.async_read_barrier();
}
