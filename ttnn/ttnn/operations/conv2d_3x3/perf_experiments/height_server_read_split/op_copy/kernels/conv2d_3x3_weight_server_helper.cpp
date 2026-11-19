// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 weight server helper, writer half (BRISC / NoC1), height_server_read_split. A helper is a second spare
// core per weight server: its NCRISC runs the server reader (conv2d_3x3_weight_server_reader.cpp) over its own part of
// every chunk into its own cb_weights (same index and size as the server's, so the same slot layout and L1 address),
// and this RISC writes each landed part into the server's slot once the server granted the round (its reader reserved
// the slot), then bumps the server's landed counter behind it. The server reader hands a chunk to the server writer
// only when its own part and every helper's part landed.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_weights = get_compile_time_arg_val(0);
    constexpr uint32_t round_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t sem_id = get_compile_time_arg_val(3);  // grant here, landed on the server
    constexpr uint32_t chunk_tiles = round_tiles / num_chunks;

    const uint32_t num_rounds = get_arg_val<uint32_t>(0);
    const uint32_t part_lo = get_arg_val<uint32_t>(1);
    const uint32_t part_tiles = get_arg_val<uint32_t>(2);
    const uint64_t server_base = get_noc_addr(get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4), 0);
    const uint64_t server_landed = server_base | get_semaphore(sem_id);
    const uint32_t tile_bytes = get_tile_size(cb_weights);
    volatile tt_l1_ptr uint32_t* grant = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    for (uint32_t r = 0; r < num_rounds; ++r) {
        {
            MaybeDeviceZoneScope("hsr_helper_grant");
            noc_semaphore_wait_min(grant, r + 1);
        }
        const uint32_t base = get_read_ptr(cb_weights);
        for (uint32_t c = 0; c < num_chunks; ++c) {
            {
                MaybeDeviceZoneScope("hsr_helper_wait");
                cb_wait_front(cb_weights, (c + 1) * chunk_tiles);
            }
            const uint32_t src = base + (c * chunk_tiles + part_lo) * tile_bytes;
            noc_async_write(src, server_base | src, part_tiles * tile_bytes);
            // The count goes out behind the part on the same NoC and VC (ordered once the part left this core).
            noc_async_writes_flushed();
            noc_semaphore_inc(server_landed, 1);
        }
        cb_pop_front(cb_weights, round_tiles);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
