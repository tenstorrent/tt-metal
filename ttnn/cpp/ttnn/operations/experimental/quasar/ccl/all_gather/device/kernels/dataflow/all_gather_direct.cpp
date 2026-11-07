// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar all-gather, DIRECT arm: one kernel of up to 6 DM threads on each rank's home cluster.
//
// Each thread pulls its packets of every slice straight from the owner's input shard into this rank's
// output shard: one NoC read of tiles_per_packet pages per packet, no staging buffer, no credits. All
// reads are issued back to back and waited for once at the end, so the in-flight depth is bounded only
// by the NoC. The destination is the local shard (the output TensorAccessor resolves the page to this
// cluster's L1), which is why this arm runs on the home cluster only.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

#include "all_gather_common.hpp"

using namespace all_gather;

void kernel_main() {
    const uint32_t rank = get_arg(args::rank);

    Noc noc;
    const auto input = TensorAccessor(tensor::input);
    const auto output = TensorAccessor(tensor::output);

    const uint32_t page_bytes = input.get_aligned_page_size();
    const uint32_t out_base = rank * pages_total;
    const ThreadPackets tp(/*worker=*/0);

    for (uint32_t s = 0; s < num_ranks; s++) {
        for (uint32_t j = tp.first; j < num_packets; j += tp.step) {
            const Packet p = packet(j);
            const uint32_t page = slice_page(s, p.k0);
            noc.async_read(input, output, p.n * page_bytes, {.page_id = page}, {.page_id = out_base + page});
        }
    }
    noc.async_read_barrier();
}
