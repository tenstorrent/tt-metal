// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// softmax writer (dim = -2, regime column_resident): store_block.
//
// Writer hart w of W drains NEOs w, w+W, ... (C/W of them) through the STRIDED cb_output_tiles ring, one tile
// each in strict rotation: the mirror of the reader. Per NEO the compute kernel packs column by column, tile-row
// by tile-row, so the k-th entry of a NEO is tile ht of its column in round r, in that order. A column a NEO
// recomputed in the padded last round is stored again to the same pages with the same bytes.
//
// Implicit sync: one tagged write per entry; no reserve, barrier, wait or pop.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t rounds = get_arg(args::rounds);
    const uint32_t num_cols = get_arg(args::num_cols);
    const uint32_t col_start = get_arg(args::col_start);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t group = C / W;

    const uint32_t w_id = get_my_thread_id();

    DataflowBuffer out(dfb::cb_output_tiles);
    Noc noc;
    const auto acc_out = TensorAccessor(tensor::output);

    auto first_page_of = [&](uint32_t neo, uint32_t round) __attribute__((always_inline)) {
        const uint32_t local = round * C + neo;
        const uint32_t u = col_start + (local < num_cols ? local : round * C);
        const uint32_t nc = u / Wt;
        return nc * Ht * Wt + (u - nc * Wt);
    };

    for (uint32_t round = 0; round < rounds; ++round) {
        uint32_t page[group];
        for (uint32_t j = 0; j < group; ++j) {
            page[j] = first_page_of(w_id + j * W, round);
        }
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            for (uint32_t j = 0; j < group; ++j) {
                noc.async_write<NocOptions::TXN_ID>(out, acc_out, {}, {.page_id = page[j] + ht * Wt});
            }
        }
    }
    // Hand the hart back with the user id in its tag register (target page, ring protocols).
    noc_async_write_set_trid(0, noc.get_noc_id());
}
