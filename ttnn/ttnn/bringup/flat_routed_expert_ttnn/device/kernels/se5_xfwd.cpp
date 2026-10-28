// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// x chain forwarder (NCRISC) on a gate/up core (MIMO_FL_X_FWD_NC, with MIMO_FL_X_CHAIN): x travels the gate/up cores
// by unicast (no multicast: the NOC0 wedge of a multicast into the gate/up rectangle beside the DRISC weight writes
// and DRAM traffic, TS_HANG_WORKLOG.md). The receiver (se5_recv.cpp, BRISC, SE_X_FWD_NC) only consumes x; this
// kernel forwards each piece as soon as it has arrived (XARR here, in pieces) into the successor's x ring, once the
// successor has freed the block's slot (SFREE here), then bumps the successor's XARR after the write is
// acknowledged; and reports min(consumed (XCONS, published by se5_recv), forwarded) blocks upstream - the
// predecessor's SFREE, or on a chain head the relay's word. It ends once se5_recv has published its last count
// (XEND = blocks + 1) and every block is forwarded and reported.
// CT: 0 X_SLOTS, 1 X_BYTES, 2 X_PIECES, 3 XARR_SEM, 4 SFREE_SEM, 5 XCONS_SEM, 6 XEND_SEM, 7 DEPTH (pieces in flight)
// RT: 0 x ring address, 1 predecessor xy (0: chain head), 2 sem id of this core's free word on the predecessor,
//     3 successor xy (0: chain tail), 4 relay xy, 5 the relay's word for this core (a chain head)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef SE_DBG
#include "api/debug/dprint.h"
#endif

void kernel_main() {
    constexpr uint32_t x_slots = get_compile_time_arg_val(0), x_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t xp = get_compile_time_arg_val(2), xarr_id = get_compile_time_arg_val(3);
    constexpr uint32_t sfree_id = get_compile_time_arg_val(4), xcons_id = get_compile_time_arg_val(5);
    constexpr uint32_t xend_id = get_compile_time_arg_val(6), depth = get_compile_time_arg_val(7);
    static_assert(depth >= 1 && depth <= 14);
    constexpr uint32_t piece = x_bytes / xp;
    const uint32_t x_ring = get_arg_val<uint32_t>(0);
    const uint32_t pxy = get_arg_val<uint32_t>(1), sxy = get_arg_val<uint32_t>(3);
    const uint64_t pred_free = get_noc_addr(pxy >> 16, pxy & 0xFFFF, get_semaphore(get_arg_val<uint32_t>(2)));
    const uint32_t rxy = get_arg_val<uint32_t>(4);
    const uint64_t relay_word = get_noc_addr(rxy >> 16, rxy & 0xFFFF, get_arg_val<uint32_t>(5));
    const uint64_t s_ring = get_noc_addr(sxy >> 16, sxy & 0xFFFF, x_ring);
    const uint64_t s_xarr = get_noc_addr(sxy >> 16, sxy & 0xFFFF, get_semaphore(xarr_id));
    auto sem = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
    volatile tt_l1_ptr uint32_t* xarr = sem(xarr_id);
    volatile tt_l1_ptr uint32_t* sfree = sem(sfree_id);
    volatile tt_l1_ptr uint32_t* xcons = sem(xcons_id);
    volatile tt_l1_ptr uint32_t* xend = sem(xend_id);
    uint32_t iss = 0, fwd = 0, rep = 0;  // pieces issued / acknowledged (+ XARR'd), blocks reported
#ifdef SE_DBG
    uint32_t iters = 0;
#endif
    while (true) {
        invalidate_l1_cache();
#ifdef SE_DBG
        if (++iters % (1u << 22) == 0) {
            DPRINT(
                "xfwd iss {} fwd {} rep {} xarr {} sfree {} cons {} end {} pxy {} sxy {}\n",
                iss,
                fwd,
                rep,
                *xarr,
                *sfree,
                *xcons,
                *xend,
                pxy,
                sxy);
        }
#endif
        if (sxy) {
            const uint32_t arrived = *xarr;
            while (iss < arrived && iss - fwd < depth && (iss / xp < x_slots || *sfree >= iss / xp + 1 - x_slots)) {
                const uint32_t off = ((iss / xp) % x_slots) * x_bytes + (iss % xp) * piece;
                for (uint32_t o = 0; o < piece; o += NOC_MAX_BURST_SIZE) {
                    const uint32_t n = piece - o < NOC_MAX_BURST_SIZE ? piece - o : NOC_MAX_BURST_SIZE;
                    noc_async_write_one_packet_with_trid(x_ring + off + o, s_ring + off + o, n, 1 + iss % depth);
                }
                ++iss;
            }
            // (the write command buffer drained first: a transaction id's outstanding count only covers requests the
            // NIU has taken - polled right after issuing, the last write can still sit in the command buffer and the
            // id reads "done" before its data has landed; the read-side twin: FLAT_COMBINE_OVERLAP_WORKLOG.md 6.14)
            while (fwd < iss && !noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
            }
            while (fwd < iss && ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, 1 + fwd % depth)) {
                noc_semaphore_inc(s_xarr, 1);
                ++fwd;
            }
        }
        const uint32_t cons = *xcons;
        const uint32_t freed = sxy && fwd / xp < cons ? fwd / xp : cons;
        if (freed > rep) {
            noc_semaphore_inc(pxy ? pred_free : relay_word, freed - rep);
            rep = freed;
        }
        const uint32_t end = *xend;
        if (end && rep == end - 1 && (!sxy || fwd == (end - 1) * xp)) {
            break;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    noc_async_write_set_trid(0);
}
