// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// x-download probe, chained column head (NCRISC, NOC1): every chunk that arrives (ARR counts them; the first head gets
// them from the readers, the others from the previous head) is multicast down this head's column and forwarded to
// the next head (unicast, up to DEPTH chunks in flight on NoC transaction ids, the next head's ARR bumped once each
// is acknowledged). Readers therefore send each byte once. At the end, bumps the column's compute cores' DONE.
// CT: 0 TOTAL_CHUNKS, 1 CHUNK_BYTES, 2 SLOTS, 3 ARR_SEM, 4 DONE_SEM
// RT: 0 ring address (heads and compute cores), 1 column start xy, 2 end xy (NOC1 order), 3 dests, 4 next head xy
//     (0: last), 5 N_DONE, 6.. xy to signal
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t total = get_compile_time_arg_val(0);
    constexpr uint32_t bytes = get_compile_time_arg_val(1);
    constexpr uint32_t slots = get_compile_time_arg_val(2);
    constexpr uint32_t arr_sem = get_compile_time_arg_val(3);
    constexpr uint32_t done_sem = get_compile_time_arg_val(4);
    constexpr uint32_t depth = 4;
    const uint32_t ring = get_arg_val<uint32_t>(0);
    const uint32_t a0 = get_arg_val<uint32_t>(1), a1 = get_arg_val<uint32_t>(2), dests = get_arg_val<uint32_t>(3);
    const uint32_t nxt = get_arg_val<uint32_t>(4);
    const uint64_t rect = get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, 0);
    const uint64_t next_ring = get_noc_addr(nxt >> 16, nxt & 0xFFFF, ring);
    const uint64_t next_arr = get_noc_addr(nxt >> 16, nxt & 0xFFFF, get_semaphore(arr_sem));
    volatile tt_l1_ptr uint32_t* arr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(arr_sem));
    auto write_trid = [](uint32_t src, uint64_t dst, uint32_t trid) {
        for (uint32_t o = 0; o < bytes; o += NOC_MAX_BURST_SIZE) {
            const uint32_t n = bytes - o < NOC_MAX_BURST_SIZE ? bytes - o : NOC_MAX_BURST_SIZE;
            noc_async_write_one_packet_with_trid(src + o, dst + o, n, trid);
        }
    };
    uint32_t mc = 0, iss = 0, fwd = 0;
    while (mc < total || (nxt && fwd < total)) {
        invalidate_l1_cache();
        const uint32_t have = *arr;
        if (mc < have) {
            noc_async_write_set_trid(0);  // the multicast must not inherit a forwarding transaction id
        }
        while (mc < have) {
            const uint32_t off = (mc % slots) * bytes;
            noc_async_write_multicast(ring + off, rect | (ring + off), bytes, dests);
            ++mc;
        }
        while (nxt && iss < have && iss - fwd < depth) {
            const uint32_t off = (iss % slots) * bytes;
            write_trid(ring + off, next_ring + off, 1 + iss % depth);
            ++iss;
        }
        while (fwd < iss && ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, 1 + fwd % depth)) {
            noc_semaphore_inc(next_arr, 1);
            ++fwd;
        }
    }
    noc_async_write_barrier();
    const uint32_t n = get_arg_val<uint32_t>(5);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t xy = get_arg_val<uint32_t>(6 + i);
        noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(done_sem)), 1);
    }
    noc_async_atomic_barrier();
}
