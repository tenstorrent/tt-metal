// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// x-download probe, column head (NCRISC, NOC1): multicasts every chunk that arrives from the readers (ARR counts them)
// down its column into the compute cores' ring, then bumps the column's compute cores' DONE.
// CT: 0 TOTAL_CHUNKS, 1 CHUNK_BYTES, 2 SLOTS, 3 ARR_SEM, 4 DONE_SEM, 5 PROBE (0 normal, 1 no multicast, 2 no wait)
// RT: 0 own ring address, 1 destination ring address, 2 column start xy, 3 end xy, 4 dests, 5 N_DONE, 6.. xy to signal
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t total = get_compile_time_arg_val(0);
    constexpr uint32_t bytes = get_compile_time_arg_val(1);
    constexpr uint32_t slots = get_compile_time_arg_val(2);
    constexpr uint32_t arr_sem = get_compile_time_arg_val(3);
    constexpr uint32_t done_sem = get_compile_time_arg_val(4);
    constexpr uint32_t probe = get_compile_time_arg_val(5);
    const uint32_t own = get_arg_val<uint32_t>(0), dst_ring = get_arg_val<uint32_t>(1);
    const uint32_t a0 = get_arg_val<uint32_t>(2), a1 = get_arg_val<uint32_t>(3), dests = get_arg_val<uint32_t>(4);
    const uint64_t rect = get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, 0);
    volatile tt_l1_ptr uint32_t* arr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(arr_sem));
    for (uint32_t i = 0; i < total; ++i) {
        if constexpr (probe != 2) {
            noc_semaphore_wait_min(arr, i + 1);
        }
        const uint32_t off = (i % slots) * bytes;
        if constexpr (probe != 1) {
            noc_async_write_multicast(own + off, rect | (dst_ring + off), bytes, dests);
        }
    }
    noc_async_write_barrier();
    if constexpr (probe == 2) {
        noc_semaphore_wait_min(arr, total);  // let the readers finish too
    }
    const uint32_t n = get_arg_val<uint32_t>(5);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t xy = get_arg_val<uint32_t>(6 + i);
        noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(done_sem)), 1);
    }
    noc_async_atomic_barrier();
}
