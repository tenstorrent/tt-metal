// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// x-download probe, self-reading relay's multicaster (NCRISC, NOC1): multicasts the chunks its BRISC reads into its
// rectangle as linked bursts (everything already read, only the last unlinked), then bumps the rectangle's compute
// cores' DONE.
// CT: 0 CB, 1 CHUNK_TILES, 2 CHUNK_BYTES, 3 TOTAL_CHUNKS, 4 SLOTS, 5 DONE_SEM
// RT: 0 ring address, 1 rect start xy, 2 end xy (NOC1 order), 3 dests, 4 N_DONE, 5.. xy
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t bytes = get_compile_time_arg_val(2);
    constexpr uint32_t total = get_compile_time_arg_val(3);
    constexpr uint32_t slots = get_compile_time_arg_val(4);
    constexpr uint32_t done_sem = get_compile_time_arg_val(5);
    const uint32_t ring = get_arg_val<uint32_t>(0);
    const uint32_t a0 = get_arg_val<uint32_t>(1), a1 = get_arg_val<uint32_t>(2), dests = get_arg_val<uint32_t>(3);
    const uint64_t rect = get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, 0);
    uint32_t c = 0;
    while (c < total) {
        uint32_t avail = 1;
        cb_wait_front(cb, chunk_tiles);
        while (c + avail < total && cb_pages_available_at_front(cb, (avail + 1) * chunk_tiles)) {
            ++avail;
        }
        const uint32_t src = get_read_ptr(cb);
        for (uint32_t i = 0; i < avail; ++i) {
            const uint32_t dst = ring + ((c + i) % slots) * bytes;
            noc_async_write_multicast(src + i * bytes, rect | dst, bytes, dests, i + 1 < avail);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb, avail * chunk_tiles);
        c += avail;
    }
    noc_async_write_barrier();
    const uint32_t n = get_arg_val<uint32_t>(4);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t xy = get_arg_val<uint32_t>(5 + i);
        noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(done_sem)), 1);
    }
    noc_async_atomic_barrier();
}
