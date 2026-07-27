// SPDX-License-Identifier: Apache-2.0
//
// Experiment 3 — full-processing RX drainer worker (tt-rdma-rx-linerate-research.md §5, exp 3). One of
// a Tensix pool that PROCESSES inbound frames (not just moves bytes like exp 2): round-robin over
// fixed-stride frames in the eth L1 ring, each worker reads a frame into local L1 (that read IS the
// compute-local landing), parses the 32B TT-RDMA header, resolves rkey -> MR (a real L1 MR-table read +
// validate), and counts processed / valid bytes. The pool's aggregate processed rate = how much RDMA
// landing throughput N Tensix workers deliver for one 200G link -> sizes the production drainer pool.
//
// Static round-robin on fixed-size frames needs NO cross-worker coordination: worker w handles frame
// indices w, w+N, w+2N, ... The DOCA test sender emits uniform jumbo frames so a fixed stride aligns.
// (A variable-size production ring needs an atomic multi-consumer head; that's a correctness layer on
// top, not a throughput change.) Landing here is compute-local (1 NoC read); a remote MR dest would add
// a second NoC write (~2x the per-frame data work) -> roughly 2x the workers.

#include <cstdint>

#include "internal/ethernet/dataflow_api.h"

void kernel_main() {
    // arg0 stats(u32[4]: bytes_lo,bytes_hi,valid,iters)  arg1 stop flag  arg2 eth src noc_x  arg3 noc_y
    // arg4 ring base  arg5 ring size  arg6 frame stride bytes  arg7 worker id  arg8 num workers
    // arg9 scratch L1 addr  arg10 MR table L1 addr (local, host-written)  arg11 mr slots
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

    volatile tt_l1_ptr uint32_t* stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_addr);
    volatile tt_l1_ptr uint32_t* stop =
        (stop_addr != 0) ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr) : nullptr;
    if (stop != nullptr) {
        *stop = 0;
    }
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);

    const uint32_t nslots = ring_size / stride;  // whole frames that fit (avoid straddle)
    uint32_t slot = wid % nslots;
    uint64_t bytes = 0;
    uint32_t valid = 0, iters = 0;
    for (;;) {
        const uint32_t off = slot * stride;
        // Land the frame into local L1 (this NoC read IS the compute-local landing).
        noc_async_read(get_noc_addr(src_x, src_y, ring_base + off), scratch, stride);
        noc_async_read_barrier();
        // Parse the 32B header from the landed copy.
        const uint32_t w0 = sc[0];
        const uint32_t len = sc[1];
        const uint32_t rkey = sc[3];
        const uint32_t op = w0 & 0xFFu;
        // Real MR-table resolve: index by rkey>>24, read the entry, validate rkey + access + bounds.
        const uint32_t mslot = rkey >> 24;
        if (mslot < mr_slots) {
            volatile tt_l1_ptr uint32_t* mr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mr_table + mslot * 32u);
            const uint32_t mr_len = mr[2];
            const uint32_t mr_rkey = mr[4];
            const uint32_t mr_access = mr[5];
            if (mr_rkey == rkey && (mr_access & 0x2u) && len <= mr_len &&
                (op == 0x10u || op == 0x11u)) {  // WRITE / WRITE_IMM
                ++valid;
            }
        }
        bytes += stride;
        slot += nworkers;
        if (slot >= nslots) {
            slot -= nslots;
        }
        if ((++iters & 0x3Fu) == 0) {
            stats[0] = (uint32_t)(bytes & 0xFFFFFFFFu);
            stats[1] = (uint32_t)(bytes >> 32);
            stats[2] = valid;
            stats[3] = iters;
            if (stop != nullptr && *stop != 0) {
                break;
            }
        }
    }
    stats[0] = (uint32_t)(bytes & 0xFFFFFFFFu);
    stats[1] = (uint32_t)(bytes >> 32);
    stats[2] = valid;
    stats[3] = iters;
}
