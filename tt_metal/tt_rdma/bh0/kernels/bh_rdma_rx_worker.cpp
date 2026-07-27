// SPDX-License-Identifier: Apache-2.0
//
// Experiment 3 — full-processing RX drainer worker (tt-rdma-rx-linerate-research.md §5, exp 3). One of
// a Tensix pool that PROCESSES inbound frames (not just moves bytes like exp 2): round-robin over
// fixed-stride frames in the eth L1 ring, each worker reads a frame into local L1 (that read IS the
// compute-local landing), parses the 32B TT-RDMA header, resolves rkey -> MR (a real L1 MR-table read +
// validate), and counts processed / valid bytes. The pool's aggregate processed rate = how much RDMA
// landing throughput N Tensix workers deliver for one 200G link -> sizes the production drainer pool.
//
// Phase 3.1b — CORRECT multi-consumer drainer (was exp3's blind round-robin). Fixed-size frames need NO
// NoC atomic: worker w statically owns frame indices {w, w+N, w+2N, ...}, so the partition is disjoint +
// complete by construction. The correctness layer added here is the **produce-head bound**: the eth
// ingest kernel publishes PKT_END_CNT (a monotonic HW count of frames fully written to L1); a worker only
// processes frame index i once i < produced, so it consumes REAL production exactly once (no reprocessing
// of unrefilled slots -> the throughput number is now honest) and never runs ahead of the MAC.
// Lapping guard: if produced outruns a worker's next index by more than the ring depth, the MAC has
// overwritten the claimed slot -> skip forward to the freshest owned index and count the gap as lapped.
// (A variable-size ring would need an atomic multi-consumer head instead; fixed frames don't.)
// Landing here is compute-local (1 NoC read); a remote MR dest (3.1c) adds a second NoC write.

#include <cstdint>

#include "internal/ethernet/dataflow_api.h"

void kernel_main() {
    // arg0 stats(u32[6]: bytes_lo,bytes_hi,valid,processed,lapped,produced)  arg1 stop  arg2 eth noc_x
    // arg3 noc_y  arg4 ring base  arg5 ring size  arg6 stride  arg7 worker id  arg8 num workers
    // arg9 scratch L1  arg10 MR table L1  arg11 mr slots  arg12 produce-head L1 addr on the eth core
    const uint32_t stats_addr = get_arg_val<uint32_t>(0);
    const uint32_t stop_addr = get_arg_val<uint32_t>(1);
    const uint32_t src_x = get_arg_val<uint32_t>(2);
    const uint32_t src_y = get_arg_val<uint32_t>(3);
    const uint32_t ring_base = get_arg_val<uint32_t>(4);
    const uint32_t ring_size = get_arg_val<uint32_t>(5);
    const uint32_t stride = get_arg_val<uint32_t>(6);
    const uint32_t wid = get_arg_val<uint32_t>(7);
    const uint32_t nworkers = get_arg_val<uint32_t>(8);
    const uint32_t scratch = get_arg_val<uint32_t>(9);
    const uint32_t mr_table = get_arg_val<uint32_t>(10);
    const uint32_t mr_slots = get_arg_val<uint32_t>(11);
    const uint32_t phead_addr = get_arg_val<uint32_t>(12);  // eth L1 addr of PKT_END_CNT (monotonic produced)

    volatile tt_l1_ptr uint32_t* stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_addr);
    volatile tt_l1_ptr uint32_t* stop =
        (stop_addr != 0) ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr) : nullptr;
    if (stop != nullptr) {
        *stop = 0;
    }
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    // Local mirror of the produce head; NoC-read from the eth core's published PKT_END_CNT.
    volatile tt_l1_ptr uint32_t* ph = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 0x1000u);

    const uint32_t nslots = ring_size / stride;  // whole frames that fit (avoid straddle)
    uint32_t next_idx = wid;                     // monotonic frame index this worker owns next
    uint32_t produced = 0;
    uint64_t bytes = 0;
    uint32_t valid = 0, processed = 0, lapped = 0, poll = 0;
    for (;;) {
        // Refresh the produce head (PKT_END_CNT) from the eth core. Cheap: one NoC read per claim batch.
        noc_async_read(get_noc_addr(src_x, src_y, phead_addr), scratch + 0x1000u, 4u);
        noc_async_read_barrier();
        produced = ph[0];

        // Lapping guard: if the MAC has advanced > (nslots - N) frames past our next index, our claimed
        // slot is already overwritten -> jump to the freshest index we own and count the skipped gap.
        if ((uint32_t)(produced - next_idx) > (nslots - nworkers)) {
            const uint32_t fresh = (produced > nslots) ? (produced - nslots) : 0u;
            uint32_t aligned = fresh + ((wid + nworkers - (fresh % nworkers)) % nworkers);  // next owned idx >= fresh
            if (aligned > next_idx) {
                lapped += (aligned - next_idx) / nworkers;
                next_idx = aligned;
            }
        }

        // Process every produced frame this worker owns, up to the head.
        while (next_idx < produced) {
            const uint32_t off = (next_idx % nslots) * stride;
            noc_async_read(get_noc_addr(src_x, src_y, ring_base + off), scratch, stride);
            noc_async_read_barrier();
            const uint32_t w0 = sc[0];
            const uint32_t len = sc[1];
            const uint32_t rkey = sc[3];
            const uint32_t op = w0 & 0xFFu;
            const uint32_t mslot = rkey >> 24;
            if (mslot < mr_slots) {
                volatile tt_l1_ptr uint32_t* mr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mr_table + mslot * 32u);
                if (mr[4] == rkey && (mr[5] & 0x2u) && len <= mr[2] && (op == 0x10u || op == 0x11u)) {
                    ++valid;
                }
            }
            bytes += stride;
            ++processed;
            next_idx += nworkers;
        }

        if ((++poll & 0x1Fu) == 0) {
            stats[0] = (uint32_t)(bytes & 0xFFFFFFFFu);
            stats[1] = (uint32_t)(bytes >> 32);
            stats[2] = valid;
            stats[3] = processed;
            stats[4] = lapped;
            stats[5] = produced;
            if (stop != nullptr && *stop != 0) {
                break;
            }
        }
    }
    stats[0] = (uint32_t)(bytes & 0xFFFFFFFFu);
    stats[1] = (uint32_t)(bytes >> 32);
    stats[2] = valid;
    stats[3] = processed;
    stats[4] = lapped;
    stats[5] = produced;
}
