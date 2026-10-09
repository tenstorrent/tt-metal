// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// x-download probe, half-grid sender (BRISC, NOC0): multicasts every chunk that arrives from the readers (ARR counts
// them) into its column run's rectangle as linked bursts: every chunk already here is sent back to back with the
// multicast path held (linked), only the last of a burst unlinked. Linked multicast runs ~67 GB/s into a 7x10
// rectangle, ~2x an unlinked one. Then bumps its rectangle's compute cores' DONE.
// CT: 0 TOTAL_CHUNKS, 1 CHUNK_BYTES, 2 SLOTS, 3 ARR_SEM, 4 DONE_SEM
// RT: 0 ring address, 1 rect start xy, 2 end xy (NOC0 order), 3 dests, 4 N_DONE, 5.. xy to signal
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t total = get_compile_time_arg_val(0);
    constexpr uint32_t bytes = get_compile_time_arg_val(1);
    constexpr uint32_t slots = get_compile_time_arg_val(2);
    constexpr uint32_t arr_sem = get_compile_time_arg_val(3);
    constexpr uint32_t done_sem = get_compile_time_arg_val(4);
#ifndef BURST
#define BURST 8
#endif
    const uint32_t ring = get_arg_val<uint32_t>(0);
    const uint32_t a0 = get_arg_val<uint32_t>(1), a1 = get_arg_val<uint32_t>(2), dests = get_arg_val<uint32_t>(3);
    const uint64_t rect = get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, 0);
    volatile tt_l1_ptr uint32_t* arr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(arr_sem));
    uint32_t mc = 0;
    while (mc < total) {
        invalidate_l1_cache();
        const uint32_t have = *arr;
        if (have - mc < BURST && have < total) {
            continue;  // wait for a full burst: a single chunk would go out unlinked
        }
        while (mc < have) {
            const uint32_t off = (mc % slots) * bytes;
#ifndef NO_MC
            noc_async_write_multicast(ring + off, rect | (ring + off), bytes, dests, mc + 1 < have);
#endif
            ++mc;
        }
    }
    noc_async_write_barrier();
    const uint32_t n = get_arg_val<uint32_t>(4);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t xy = get_arg_val<uint32_t>(5 + i);
        noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(done_sem)), 1);
    }
    noc_async_atomic_barrier();
}
