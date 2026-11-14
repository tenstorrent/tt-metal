// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar all-gather, RING arm writer: T threads, CONSUMER of the packet DFB. Thread t drains exactly what
// reader thread t produced (equal thread counts, STRIDED both sides). One DFB entry is one packet, so a
// packet leaves as ONE NoC write per destination.
//
// Hop 0: own slice -> this rank's shard AND the next rank's shard, then a ready credit to the next rank's
// matching (worker, thread) semaphore every chunks_per_sync packets. Hops 1..N-3: forward the re-read
// slice to the next rank's shard with the same credits.
//
// A DFB entry is released once its data has left L1 (async_writes_flushed), not once the remote write is
// acknowledged; the acknowledgements are collected once at the end. The ready credit also follows a
// flush, so it cannot overtake the payload. The transport seam is the remote write + send_credit():
// NoC write + Semaphore::up today.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

#include "all_gather_common.hpp"

using namespace all_gather;

void kernel_main() {
    const uint32_t rank = get_arg(args::rank);
    const uint32_t worker = get_arg(args::worker);
    const uint32_t next_noc_x = get_arg(args::next_noc_x);
    const uint32_t next_noc_y = get_arg(args::next_noc_y);

    Noc noc;
    DataflowBuffer pkt(dfb::pkt);
    const auto output = TensorAccessor(tensor::output);
    Semaphore ready_sem(ready_token(get_my_thread_id()));

    const uint32_t page_bytes = output.get_aligned_page_size();
    const uint32_t out_base = rank * pages_total;
    const uint32_t next_base = ((rank + 1) % num_ranks) * pages_total;
    const ThreadPackets tp(worker);

    auto send_credit = [&]() {
        noc.async_writes_flushed();
        ready_sem.up(noc, next_noc_x, next_noc_y, 1);
    };

    // Hop 0 (own slice -> local + next) and hops 1..N-3 (forward); the slice at hop h came from rank-h.
    for (uint32_t hop = 0; hop + 1 < num_ranks; hop++) {
        const uint32_t s = (rank + num_ranks - hop) % num_ranks;
        uint32_t packets_sent = 0;
        for (uint32_t j = tp.first; j < num_packets; j += tp.step) {
            const Packet p = packet(j);
            const uint32_t page = slice_page(s, p.k0);
            const uint32_t bytes = p.n * page_bytes;
            pkt.wait_front(1);
            if (hop == 0) {
                noc.async_write(pkt, output, bytes, {.offset_bytes = 0}, {.page_id = out_base + page});
            }
            noc.async_write(pkt, output, bytes, {.offset_bytes = 0}, {.page_id = next_base + page});
            noc.async_writes_flushed();
            pkt.pop_front(1);
            packets_sent++;
            if (packets_sent % chunks_per_sync == 0) {
                send_credit();
            }
        }
        if (packets_sent % chunks_per_sync != 0) {
            send_credit();
        }
    }

    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
