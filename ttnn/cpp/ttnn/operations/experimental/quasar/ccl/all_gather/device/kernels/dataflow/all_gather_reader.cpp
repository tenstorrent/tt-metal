// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar all-gather, RING arm reader: T threads, PRODUCER of the packet DFB. One DFB entry holds one
// packet (tiles_per_packet pages, contiguous), so every packet is ONE NoC read.
//
// Hop 0 reads this rank's own slice from the input shard. Hops 1..N-1 wait for the previous rank's ready
// credits on this thread's semaphore, then (hops 1..N-2) re-read the landed slice from this rank's output
// shard (store-and-forward) so the writer can push it on. The last received slice (originally the next
// rank's own) is only waited for, never forwarded.
//
// The ring arm is gated by a preprocessor define, not if-constexpr: the sem:: tokens exist only when the
// host bound the semaphores, and a discarded if-constexpr branch still has to name them.

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

namespace {

// Pull packet j of slice s from `src` (page ids page_base + slice_page(s, .)) into one DFB entry.
template <typename Dfb, typename Accessor>
FORCE_INLINE void read_packet(
    Noc& noc, Dfb& pkt, const Accessor& src, uint32_t page_base, uint32_t s, uint32_t j, uint32_t page_bytes) {
    const Packet p = packet(j);
    pkt.reserve_back(1);
    noc.async_read(src, pkt, p.n * page_bytes, {.page_id = page_base + slice_page(s, p.k0)}, {.offset_bytes = 0});
    noc.async_read_barrier();
    pkt.push_back(1);
}

}  // namespace

void kernel_main() {
    const uint32_t rank = get_arg(args::rank);
    const uint32_t worker = get_arg(args::worker);

    Noc noc;
    DataflowBuffer pkt(dfb::pkt);
    const auto input = TensorAccessor(tensor::input);
    const auto output = TensorAccessor(tensor::output);
    Semaphore ready_sem(ready_token(get_my_thread_id()));

    const uint32_t page_bytes = input.get_aligned_page_size();
    const ThreadPackets tp(worker);

    // Hop 0: own slice from the input shard.
    for (uint32_t j = tp.first; j < num_packets; j += tp.step) {
        read_packet(noc, pkt, input, 0, rank, j, page_bytes);
    }

    // Hops 1..N-1: slices arriving from the previous rank, oldest origin first: rank-1, rank-2, ...
    // The sender credits ready<t> once per chunks_per_sync packets of this thread's share (plus once for a
    // short tail); this side waits for credit k+1 before packet k*chunks_per_sync, cumulatively over hops.
    const uint32_t out_base = rank * pages_total;
    uint32_t credits_waited = 0;
    for (uint32_t hop = 1; hop < num_ranks; hop++) {
        const uint32_t s = (rank + num_ranks - hop) % num_ranks;
        const bool forward = hop < num_ranks - 1;
        uint32_t packets_seen = 0;
        for (uint32_t j = tp.first; j < num_packets; j += tp.step) {
            if (packets_seen % chunks_per_sync == 0) {
                credits_waited++;
                ready_sem.wait_min(credits_waited);
            }
            packets_seen++;
            if (forward) {
                read_packet(noc, pkt, output, out_base, s, j, page_bytes);
            }
        }
    }
}
