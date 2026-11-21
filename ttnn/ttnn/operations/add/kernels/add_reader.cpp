// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// add reader (load_block): this cluster's out pages [start_id, start_id + num_tiles); a's pages into its ring
// and the b tiles they need into b's ring.
//
// Implicit sync (the default protocol of a NoC-filled ring on Quasar): one read tagged NocOptions::TXN_ID per
// page lands at the ring's cursor and the DM0 interrupt handler posts its credit, so there is no reserve,
// barrier or push, and the hart keeps as many reads in flight as the ring has free entries.
//
// Regimes (`bcast_mode`, add_bcast_walk.hpp):
//   FLAT / LEADING  page p of a and b's page of p (p itself, or remapped over b's broadcast leading dimensions).
//                   Thread plan: reader hart r of R reads pages r, r+R, ...; the STRIDED rings deal them.
//   COL/ROW/SCALAR  the cluster's runs: each b window once, when it opens (explicit sync; it stays resident in
//                   b's ring until compute pops it), then the run's a pages (implicit sync). Plan (1, 1, 1).
//   NUMBER          a only (ADD_B_IS_NUMBER: no b tensor, no b ring).
//
// No dataflow helper covers a tensor-page -> ring transfer (op_design.md, "Helpers considered and rejected").

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

#include "add_bcast_walk.hpp"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t mode = get_arg(args::bcast_mode);

    DataflowBuffer cb_input_a_tiles(dfb::cb_input_a_tiles);
    Noc noc;
    const auto acc_a = TensorAccessor(tensor::input_a);

#ifdef ADD_B_IS_NUMBER
    static_assert(mode == add_bcast::MODE_NUMBER, "ADD_B_IS_NUMBER goes with the number regime");
    for (uint32_t i = get_my_thread_id(); i < num_tiles; i += R) {
        noc.async_read<NocOptions::TXN_ID>(acc_a, cb_input_a_tiles, {.page_id = start_id + i}, {});
    }
#else
    DataflowBuffer cb_input_b_tiles(dfb::cb_input_b_tiles);
    const auto acc_b = TensorAccessor(tensor::input_b);
    const add_bcast::Geometry geom{
        get_arg(args::Ht), get_arg(args::Wt), get_arg(args::n_inner), get_arg(args::s_outer), get_arg(args::s_inner)};

    if constexpr (mode == add_bcast::MODE_FLAT || mode == add_bcast::MODE_LEADING) {
        const uint32_t img = geom.Ht * geom.Wt;
        for (uint32_t i = get_my_thread_id(); i < num_tiles; i += R) {
            const uint32_t p = start_id + i;
            const uint32_t b_page = mode == add_bcast::MODE_FLAT ? p : geom.nb(p / img) * img + p % img;
            noc.async_read<NocOptions::TXN_ID>(acc_a, cb_input_a_tiles, {.page_id = p}, {});
            noc.async_read<NocOptions::TXN_ID>(acc_b, cb_input_b_tiles, {.page_id = b_page}, {});
        }
    } else {
        // b's ring is explicit sync (the host opts it out of implicit sync): a window is reserved, read with the
        // user transaction id, barriered on that id and pushed at once, so compute sees it as soon as it lands.
        // Under implicit sync its credit would wait for a whole batch (half the ring) of b reads, which can sit
        // behind a reads this hart cannot issue until compute frees a's ring: a deadlock. a stays implicit.
        constexpr bool uniform = get_arg(args::uniform_windows) != 0;
        constexpr NocOptVals user_txn = {.trid = 0};
        const uint32_t b_tile_bytes = cb_input_b_tiles.get_tile_size();
        add_bcast::for_each_run<mode>(geom, start_id, num_tiles, uniform, [&](const add_bcast::Run& run) {
            if (run.open) {
                cb_input_b_tiles.reserve_back(run.win_tiles);
                for (uint32_t k = 0; k < run.win_tiles; ++k) {
                    noc.async_read<NocOptions::TXN_ID>(
                        acc_b,
                        cb_input_b_tiles,
                        b_tile_bytes,
                        {.page_id = run.win_first + k},
                        {.offset_bytes = k * b_tile_bytes},
                        user_txn);
                }
                noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
                cb_input_b_tiles.push_back(run.win_tiles);
            }
            for (uint32_t k = 0; k < run.len; ++k) {
                noc.async_read<NocOptions::TXN_ID>(acc_a, cb_input_a_tiles, {.page_id = run.first_page + k}, {});
            }
        });
    }
#endif
    // No final barrier: firmware credits the last partial batch and drains the rings when the kernel returns.
    // Hand the hart back with the user id in its (sticky) tag register.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
