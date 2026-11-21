// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// sigmoid reader (load_block): reader thread p of R reads this cluster's pages p, p+R, ... of its contiguous
// run [start_id, start_id + num_tiles) into cb_input_tiles. The STRIDED ring deals entry k to NEO k mod C, so
// each page lands in the slice of the NEO that computes it.
//
// Implicit sync: one transaction-tagged read per entry; the DM0 interrupt handler posts the credits, so there
// is no reserve / barrier / push. The tag register is handed back as the last statement (target page, ring
// protocols): a tt-metal program that runs next with untagged transfers would otherwise be counted for a ring.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);
    constexpr uint32_t R = get_arg(args::reader_threads);

    DataflowBuffer cb_input_tiles(dfb::cb_input_tiles);
    Noc noc;
    const auto input_accessor = TensorAccessor(tensor::input);

    for (uint32_t i = get_my_thread_id(); i < num_tiles; i += R) {
        noc.async_read<NocOptions::TXN_ID>(input_accessor, cb_input_tiles, {.page_id = start_id + i}, {});
    }
    // No final barrier: firmware credits the last batch and drains the ring when the kernel returns.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
