// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// permute reader (page_gather and tile_transpose): load_block over this cluster's run of output pages.
//
// Output page p of the run (output page order) reads input page src(p) = sum_k c_k * st[k], where (c0..c3) is
// p's coordinate on the output page grid g[0..3]. The host gives the run's first coordinate; an odometer steps
// it, so there is no division per page. The page lands in the ring with implicit sync: one tagged read per
// entry, the DM0 interrupt handler posts the credits.
//
// Reader thread r of R reads the run's pages r, r+R, ...: on a STRIDED ring its k-th entry is the run's page
// r + k*R, which is the entry the consumer (writer or NEO) expects in that slot.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    // The plan's count, not get_num_threads(): a thread that did not run leaves its pages missing.
    constexpr uint32_t R = get_arg(args::reader_threads);

    const uint32_t g[4] = {get_arg(args::g0), get_arg(args::g1), get_arg(args::g2), get_arg(args::g3)};
    const uint32_t st[4] = {get_arg(args::st0), get_arg(args::st1), get_arg(args::st2), get_arg(args::st3)};
    uint32_t c[4] = {get_arg(args::c0), get_arg(args::c1), get_arg(args::c2), get_arg(args::c3)};

    DataflowBuffer ring(dfb::ring);
    Noc noc;
    const auto acc_in = TensorAccessor(tensor::input);

    uint32_t src = c[0] * st[0] + c[1] * st[1] + c[2] * st[2] + c[3] * st[3];
    uint32_t turn = 0;  // == page index mod R
    const uint32_t me = get_my_thread_id();
    for (uint32_t i = 0; i < num_pages; ++i) {
        if (turn == me) {
            noc.async_read<NocOptions::TXN_ID>(acc_in, ring, {.page_id = src}, {});
        }
        if (++turn == R) {
            turn = 0;
        }
        // Odometer: next output page, last grid axis fastest.
        for (int k = 3; k >= 0; --k) {
            src += st[k];
            if (++c[k] < g[k]) {
                break;
            }
            src -= g[k] * st[k];
            c[k] = 0;
        }
    }
    // No barrier: firmware credits the last partial batch when the kernel returns. Hand the sticky transaction
    // tag back so an untagged transfer of the next program is not counted for this ring.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
