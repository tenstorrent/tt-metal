// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// multiply reader. One of two regimes, chosen on the host (CT arg `regime`):
//
// flat_stream (regime 0, equal shapes): a and b have the output's page ids. Hart p of R reads this
//   cluster's tiles p, p+R, ... of each operand, implicit sync (tagged reads, the DM0 ISR posts credits).
//
// row_bcast_strips (regime 1, b of size 1 along H): per block of the shared walker,
//   load_b_window  the block's `cols` b tiles into one nominal `block_cols` window of the b ring, explicit
//                  sync (reserve / user-trid reads / one barrier / push); the window stays resident while
//                  compute walks the block's rows. Nominal windows keep every window aligned to the ring, so
//                  compute's indexed reads never straddle the ring's end.
//   load_a_block   the block's rows x cols a tiles row-major into the a ring, implicit sync.
//   b's image is mapped through (sb0, sb1), so a b broadcast along leading dims is read from its one image.
//
// The kernel's last transfer is a ring-tagged a read in both regimes, so it hands the tag register back
// (target page, ring protocols).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "multiply_block_walker.hpp"

void kernel_main() {
    constexpr uint32_t regime = get_arg(args::regime);
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t block_cols = get_arg(args::block_cols);
    const uint32_t start = get_arg(args::start);
    const uint32_t count = get_arg(args::count);

    DataflowBuffer dfb_a(dfb::in_a);
    DataflowBuffer dfb_b(dfb::in_b);
    Noc noc;
    const auto acc_a = TensorAccessor(tensor::a);
    const auto acc_b = TensorAccessor(tensor::b);

    if constexpr (regime == 0) {
        // load_stream: the core's whole run, dealt over the R reader harts.
        for (uint32_t i = get_my_thread_id(); i < count; i += R) {
            noc.async_read<NocOptions::TXN_ID>(acc_a, dfb_a, {.page_id = start + i}, {});
            noc.async_read<NocOptions::TXN_ID>(acc_b, dfb_b, {.page_id = start + i}, {});
        }
    } else {
        static_assert(regime == 0 || R == 1, "row_bcast_strips deals whole blocks: one reader hart");
        const uint32_t G = get_arg(args::G);
        const uint32_t Ht = get_arg(args::Ht);
        const uint32_t Wt = get_arg(args::Wt);
        const uint32_t o1 = get_arg(args::o1);
        const uint32_t sb0 = get_arg(args::sb0);
        const uint32_t sb1 = get_arg(args::sb1);
        const uint32_t tile_bytes = dfb_b.get_tile_size();
        constexpr NocOptVals user_txn = {.trid = 0};

        multiply::BlockWalker walker(start, count, G, Ht, Wt, block_cols);
        multiply::Block blk;
        while (walker.next(blk)) {
            // load_b_window: b has one tile row per image.
            const uint32_t d0 = blk.n / o1;
            const uint32_t d1 = blk.n - d0 * o1;
            const uint32_t b_base = (d0 * sb0 + d1 * sb1) * Wt + blk.col_begin;
            dfb_b.reserve_back(block_cols);
            for (uint32_t j = 0; j < blk.cols; ++j) {
                noc.async_read<NocOptions::TXN_ID>(
                    acc_b, dfb_b, tile_bytes, {.page_id = b_base + j}, {.offset_bytes = j * tile_bytes}, user_txn);
            }
            noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
            dfb_b.push_back(block_cols);

            // load_a_block: rows x cols, row-major.
            uint32_t a_row = (blk.n * Ht + blk.row_begin) * Wt + blk.col_begin;
            for (uint32_t r = 0; r < blk.rows; ++r, a_row += Wt) {
                for (uint32_t j = 0; j < blk.cols; ++j) {
                    noc.async_read<NocOptions::TXN_ID>(acc_a, dfb_a, {.page_id = a_row + j}, {});
                }
            }
        }
    }
    // Hand the hart back with the user id in its tag register: the last read carried a ring's id, and a
    // program that follows with untagged transfers would have them counted for that ring.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
