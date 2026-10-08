// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// simple_add writer: writes tile i of dfb::out to page i of the DRAM-interleaved output tensor C. With
// implicit_sync (Quasar, chosen by the host) each write is tagged with a DFB transaction id and the DM0 ISR acks
// the tile-counter credits once the writes are sent, so there is no wait_front/barrier/pop_front. Otherwise the
// kernel does the explicit sequence. Runs as N threads, one per DM core (2 on Quasar, 1 on Wormhole/Blackhole): thread
// t writes tiles t, t+N, t+2N, ... Each pop_front rotates to the thread's next tile counter, so tile i comes from
// compute thread i % num_compute_threads. With N == 2 and 4 Tensix, DM t alternates between Tensix t and t + 2.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    [[maybe_unused]] constexpr bool implicit_sync = get_arg(args::implicit_sync) != 0;

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);
    const uint32_t out_tile_bytes = dfb_out.get_entry_size();

    const auto c = TensorAccessor(tensor::c);

    const uint32_t num_threads = get_num_threads();
    for (uint32_t i = get_my_thread_id(); i < num_tiles; i += num_threads) {
#ifdef ARCH_QUASAR
        if constexpr (implicit_sync) {
            // Waits for a tile on this thread's current tile counter, writes it out and moves to the next counter;
            // the ack is posted by the ISR when the write is sent.
            noc.async_write<NocOptions::TXN_ID>(dfb_out, c, {}, {.page_id = i});
            continue;
        }
#endif
        dfb_out.wait_front(1);
        noc.async_write(dfb_out, c, out_tile_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out.pop_front(1);
    }

    // With implicit sync, also acks any partial transaction-id batch; on Quasar ~DataflowBuffer() then waits for
    // the writes to land.
    dfb_out.finish();
}
