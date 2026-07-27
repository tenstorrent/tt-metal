// SPDX-License-Identifier: Apache-2.0
//
// Experiment 2 — RX drain-half bandwidth probe (tt-rdma-rx-linerate-research.md §5.2). One Tensix
// worker of a pool: continuously NoC-reads fixed-size chunks OUT of the eth core's L1 RX ring into
// local scratch, counting bytes moved. The pool's aggregate read rate = how fast the eth-core L1 can
// be DRAINED. Run concurrently with the MAC filling that same L1 at line rate (the DOCA sender + the
// ingest probe on the eth core) to answer: can one eth L1 sustain ~200G write + ~200G read at once
// (the mandatory double-copy for RX-to-MR), or is the L1 read port the single-link-200G ceiling?
//
// This is a bandwidth probe: it reads possibly-stale/overwritten ring bytes on purpose (we measure
// L1/NoC throughput, not data correctness). Each worker sweeps a strided region so the pool covers the
// whole ring rather than hammering one line.

#include <cstdint>

#include "internal/ethernet/dataflow_api.h"  // noc_async_read, get_noc_addr, noc_async_read_barrier

void kernel_main() {
    // arg0 stats(bytes-moved lo/hi + iters) L1 addr   arg1 stop flag L1 addr
    // arg2 eth src noc_x   arg3 eth src noc_y   arg4 eth ring base   arg5 ring size bytes
    // arg6 my start offset   arg7 chunk bytes   arg8 stride bytes (pool-wide)   arg9 local scratch L1 addr
    const uint32_t stats_addr = get_arg_val<uint32_t>(0);
    const uint32_t stop_addr = get_arg_val<uint32_t>(1);
    const uint32_t src_x = get_arg_val<uint32_t>(2);
    const uint32_t src_y = get_arg_val<uint32_t>(3);
    const uint32_t ring_base = get_arg_val<uint32_t>(4);
    const uint32_t ring_size = get_arg_val<uint32_t>(5);
    const uint32_t my_off0 = get_arg_val<uint32_t>(6);
    const uint32_t chunk = get_arg_val<uint32_t>(7);
    const uint32_t stride = get_arg_val<uint32_t>(8);
    const uint32_t scratch = get_arg_val<uint32_t>(9);

    volatile tt_l1_ptr uint32_t* stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_addr);
    volatile tt_l1_ptr uint32_t* stop =
        (stop_addr != 0) ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr) : nullptr;
    if (stop != nullptr) {
        *stop = 0;
    }

    uint32_t off = my_off0 % ring_size;
    uint64_t bytes = 0;
    uint32_t iters = 0;
    for (;;) {
        // Read one chunk out of the eth L1 ring (straddle-safe: keep chunk aligned + ring_size % chunk == 0
        // caller-guaranteed, so a chunk never wraps).
        noc_async_read(get_noc_addr(src_x, src_y, ring_base + off), scratch, chunk);
        noc_async_read_barrier();
        bytes += chunk;
        off += stride;
        if (off >= ring_size) {
            off -= ring_size;
        }
        if ((++iters & 0x3Fu) == 0) {  // publish periodically (cheap)
            stats[0] = (uint32_t)(bytes & 0xFFFFFFFFu);
            stats[1] = (uint32_t)(bytes >> 32);
            stats[2] = iters;
            if (stop != nullptr && *stop != 0) {
                break;
            }
        }
    }
    stats[0] = (uint32_t)(bytes & 0xFFFFFFFFu);
    stats[1] = (uint32_t)(bytes >> 32);
    stats[2] = iters;
}
