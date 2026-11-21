// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// scaled_dot_product_attention writer: hart w of W writer harts of a cluster drains NEOs w, w+W, ... (C/W of
// them) through the STRIDED `out` ring, one tile each in strict rotation. Per round every NEO pushes its unit's
// q_chunk_tiles x Dt output tiles row-major; a padded NEO (or a clamped row) rewrites a page another NEO (row)
// writes too, with the same bytes.
//
// Implicit sync: one tagged write per ring entry, the DM0 interrupt handler acks the entries. The hart hands
// its tag register back as its last statement (a tt-metal program that runs next issues untagged transfers).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

#include "sdpa_geometry.hpp"

void kernel_main() {
    const uint32_t first_round = get_arg(args::first_round);
    const uint32_t num_rounds = get_arg(args::num_rounds);
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t Dt = get_arg(args::Dt);
    constexpr uint32_t qc = get_arg(args::q_chunk_tiles);
    constexpr uint32_t neos_per_hart = C / W;
    using G = SDPA_GEOMETRY_FROM_ARGS;
    static_assert(C % W == 0, "W must divide C");

    const uint32_t w = get_my_thread_id();
    DataflowBuffer out(dfb::out);
    Noc noc;
    const auto acc_out = TensorAccessor(tensor::out);

    for (uint32_t r = 0; r < num_rounds; ++r) {
        const auto rc = G::round_coords(first_round + r);
        for (uint32_t i = 0; i < qc; ++i) {
            for (uint32_t d = 0; d < Dt; ++d) {
                for (uint32_t j = 0; j < neos_per_hart; ++j) {
                    const uint32_t u = G::unit_of(rc.head_round, w + j * W);
                    const uint32_t page = G::q_page(rc.b, G::q_head(rc.c, u), G::q_row(u, i), d);
                    noc.async_write<NocOptions::TXN_ID>(out, acc_out, {}, {.page_id = page});
                }
            }
        }
    }
    // Firmware waits for the tagged writes to land when the kernel returns; hand the tag register back.
    noc_async_write_set_trid(0, noc.get_noc_id());
}
