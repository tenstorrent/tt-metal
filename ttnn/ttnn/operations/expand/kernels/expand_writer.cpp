// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// expand writer (regime R-H): store_fanout_block.
//
// Same blocks as the reader (source tiles s_first..s_last of this cluster's run, block_src_tiles at a time).
// Per block: one wait for the nominal block; entry k (source s = s0 + k = p * tensor_w_tiles + j) is written to
// every output tile of its row segment [r_begin, r_end) that lies in this cluster's run, at output page
// p * fan_out_tiles * tensor_w_tiles + r * tensor_w_tiles + j; one write barrier; one pop. The barrier comes
// before the pop: the entry must not be handed back to the reader while a write out of it is in flight.
//
// No helper covers the step: these are DRAM writes through the TensorAccessor, one per output page, all out
// of one resident entry (kernel_lib's dataflow copy helpers are L1->L1 only, its mcast helpers core-to-core).
//
// Explicit sync: the host opts the ring out of implicit sync, and every write carries the user transaction
// id 0 (see the reader).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t run_start = get_arg(args::run_start);
    const uint32_t run_tiles = get_arg(args::run_tiles);
    constexpr uint32_t block_src_tiles = get_arg(args::block_src_tiles);
    constexpr uint32_t fan_out_tiles = get_arg(args::fan_out_tiles);
    constexpr uint32_t tensor_w_tiles = get_arg(args::tensor_w_tiles);
    constexpr uint32_t W = get_arg(args::writer_threads);
    static_assert(W == 1, "regime R-H deals blocks to one writer thread");
    constexpr NocOptVals user_txn = {.trid = 0};
    // A transaction id counts its outstanding transfers in 16 bits: drain before that many are in flight.
    constexpr uint32_t max_writes_in_flight = 0x8000;

    DataflowBuffer ring(dfb::cb_row_bcast_tiles);
    Noc noc;
    const auto acc_out = TensorAccessor(tensor::output);
    const uint32_t page_size = ring.get_tile_size();

    if (run_tiles == 0) {
        return;
    }
    const uint32_t run_end = run_start + run_tiles;
    const uint32_t s_first = run_start / fan_out_tiles;
    const uint32_t s_end = (run_end - 1) / fan_out_tiles + 1;
    uint32_t in_flight = 0;

    for (uint32_t s0 = s_first; s0 < s_end; s0 += block_src_tiles) {
        const uint32_t n = (s_end - s0 < block_src_tiles) ? (s_end - s0) : block_src_tiles;
        ring.wait_front(block_src_tiles);
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t s = s0 + k;
            const uint32_t seg_start = s * fan_out_tiles;  // L of this source's row 0
            const uint32_t r_begin = (run_start > seg_start) ? run_start - seg_start : 0;
            const uint32_t r_end = (run_end < seg_start + fan_out_tiles) ? run_end - seg_start : fan_out_tiles;
            const uint32_t p = s / tensor_w_tiles;
            const uint32_t j = s - p * tensor_w_tiles;
            uint32_t page = p * fan_out_tiles * tensor_w_tiles + r_begin * tensor_w_tiles + j;
            for (uint32_t r = r_begin; r < r_end; ++r, page += tensor_w_tiles) {
                noc.async_write<NocOptions::TXN_ID>(
                    ring, acc_out, page_size, {.offset_bytes = k * page_size}, {.page_id = page}, user_txn);
                if (++in_flight == max_writes_in_flight) {
                    noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
                    in_flight = 0;
                }
            }
        }
        noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
        in_flight = 0;
        ring.pop_front(block_src_tiles);
    }
    // Hand the hart back with the user id in its sticky tag register.
    noc_async_write_set_trid(0, noc.get_noc_id());
}
