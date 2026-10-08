// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// simple_add reader: reads tile i of A into dfb::in0 and tile i of B into dfb::in1 from DRAM-interleaved
// tensors. With implicit_sync (Quasar, chosen by the host) each read is tagged with a DFB transaction id and the
// DM0 ISR posts the tile-counter credits once the reads land, so there is no reserve_back/barrier/push_back.
// Otherwise (Wormhole/Blackhole, or a Quasar shape the host keeps on explicit sync) the kernel does the explicit
// sequence. Runs as N threads, one per DM core (4 on
// Quasar, 1 on Wormhole/Blackhole): thread t reads tiles t, t+N, t+2N, ... Each push_back rotates to the thread's next
// tile counter, so tile i still goes to compute thread i % num_compute_threads. With N == 4 and 4 Tensix each
// thread has one tile counter, so DM t feeds only Tensix t.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    // num_input_tiles >= num_tiles: tiles past num_tiles are filler that keeps every implicit-sync batch full.
    const uint32_t num_input_tiles = get_arg(args::num_input_tiles);
    [[maybe_unused]] constexpr bool implicit_sync = get_arg(args::implicit_sync) != 0;

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    const uint32_t in0_tile_bytes = dfb_in0.get_entry_size();
    const uint32_t in1_tile_bytes = dfb_in1.get_entry_size();

    const auto a = TensorAccessor(tensor::a);
    const auto b = TensorAccessor(tensor::b);

    const uint32_t num_threads = get_num_threads();
    for (uint32_t i = get_my_thread_id(); i < num_input_tiles; i += num_threads) {
        const uint32_t page = i < num_tiles ? i : 0;  // filler tiles re-read page 0; compute discards them
#ifdef ARCH_QUASAR
        if constexpr (implicit_sync) {
            // Waits for space on this thread's current tile counter, reads one entry into it and moves to the next
            // counter; the credit is posted by the ISR when the read completes.
            noc.async_read<NocOptions::TXN_ID>(a, dfb_in0, {.page_id = page}, {});
            noc.async_read<NocOptions::TXN_ID>(b, dfb_in1, {.page_id = page}, {});
            continue;
        }
#endif
        dfb_in0.reserve_back(1);
        dfb_in1.reserve_back(1);
        noc.async_read(a, dfb_in0, in0_tile_bytes, {.page_id = page}, {});
        noc.async_read(b, dfb_in1, in1_tile_bytes, {.page_id = page}, {});
        noc.async_read_barrier();
        dfb_in0.push_back(1);
        dfb_in1.push_back(1);
    }

    // With implicit sync, also posts any partial transaction-id batch the ISR has not. Waits for compute to ack
    // every tile.
    dfb_in0.finish();
    dfb_in1.finish();
}
