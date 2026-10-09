// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Flat spatial expert: x relay multicaster (NCRISC, NOC1) on a core just outside its rectangle of gate/up cores. Its
// BRISC reads x blocks from DRAM (xdl_selfread.cpp); this kernel multicasts them into the rectangle's x rings as
// linked bursts (every block already read and whose ring slot every core of the rectangle has freed), then sets the
// cores' XARR to the number of blocks delivered with a multicast that follows the last linked block (so it lands
// after the data without waiting for acks). A core's freed count lives in its own word of FREED_WORDS here (the
// minimum over the rectangle gates slot reuse: a summed count would let fast cores stand in for a slow one).
// Linked multicast into a 7-wide rectangle runs ~67 GB/s; the relay reading on NOC0 and multicasting on NOC1 keeps
// the two traffic kinds apart.
// CT: 0 CB, 1 BLOCK_TILES, 2 BLOCK_BYTES, 3 TOTAL_BLOCKS, 4 X_SLOTS, 5 XARR_SEM, 6 CB_BLOCKS (blocks the local CB
// holds: a burst stops at its wrap), 7 WORD_SEM
//     (source word of the XARR multicast)
// RT: 0 x ring address, 1 rect start xy, 2 end xy (NOC1 order), 3 dests, 4 freed words address, 5 cores in the
// rectangle (freed words)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef SE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#endif

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t bytes = get_compile_time_arg_val(2);
    constexpr uint32_t total = get_compile_time_arg_val(3);
    constexpr uint32_t x_slots = get_compile_time_arg_val(4);
    constexpr uint32_t xarr_sem = get_compile_time_arg_val(5);
    constexpr uint32_t cb_blocks = get_compile_time_arg_val(6);
    constexpr uint32_t word_sem = get_compile_time_arg_val(7);
    const uint32_t ring = get_arg_val<uint32_t>(0);
    const uint32_t a0 = get_arg_val<uint32_t>(1), a1 = get_arg_val<uint32_t>(2), dests = get_arg_val<uint32_t>(3);
    const uint32_t n_cores = get_arg_val<uint32_t>(5);
    volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(4));
    const uint64_t rect = get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, 0);
    volatile tt_l1_ptr uint32_t* word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(word_sem));
    const uint32_t xarr_addr = get_semaphore(xarr_sem);
    auto min_freed = [&]() {
        uint32_t lo = freed[0];
        for (uint32_t i = 1; i < n_cores; ++i) {
            lo = freed[i] < lo ? freed[i] : lo;
        }
        return lo;
    };
    uint32_t sent = 0;
    while (sent < total) {
        invalidate_l1_cache();
        // blocks ready (read) and whose slot is free everywhere
        uint32_t avail = 0;
        uint32_t lim = min_freed() + x_slots;
        const uint32_t wrap = sent + cb_blocks - sent % cb_blocks;  // blocks contiguous from the read pointer
        lim = wrap < lim ? wrap : lim;
        while (sent + avail < total && sent + avail < lim && cb_pages_available_at_front(cb, (avail + 1) * blk)) {
            ++avail;
        }
        if (!avail) {
#ifdef SE_ZONES
            if (sent < lim) {
                DeviceZoneScopedN("XMC_WAIT_CB");
                cb_wait_front(cb, blk);
            } else {
                DeviceZoneScopedN("XMC_WAIT_CREDIT");
                do {
                    invalidate_l1_cache();
                } while (min_freed() + x_slots <= sent);
            }
#endif
            continue;
        }
        const uint32_t src = get_read_ptr(cb);
        for (uint32_t i = 0; i < avail; ++i) {
            const uint32_t dst = ring + ((sent + i) % x_slots) * bytes;
#ifdef XMC_SAFE
            noc_async_write_multicast(src + i * bytes, rect | dst, bytes, dests, i + 1 < avail);
#else
            noc_async_write_multicast(src + i * bytes, rect | dst, bytes, dests, true);  // linked to what follows
#endif
        }
        sent += avail;
#ifdef XMC_SAFE
        noc_async_write_barrier();
#endif
        while (!ncrisc_noc_nonposted_writes_sent(noc_index)) {
        }  // the previous counter write has left L1 before its source word is reused
        *word = sent;
        noc_semaphore_set_multicast(reinterpret_cast<uint32_t>(word), rect | xarr_addr, dests);
        noc_async_writes_flushed();
        cb_pop_front(cb, avail * blk);
    }
    noc_async_write_barrier();
}
