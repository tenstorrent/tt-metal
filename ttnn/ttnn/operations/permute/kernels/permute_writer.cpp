// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// permute writer (page_gather and tile_transpose): store_block over this cluster's run of output pages,
// consecutive page ids from start_page. Writer thread w of W writes the run's pages w, w+W, ... (its entries of
// the STRIDED ring). Implicit sync: one tagged write per entry, the DM0 interrupt handler acks it.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_page = get_arg(args::start_page);
    constexpr uint32_t W = get_arg(args::writer_threads);

    DataflowBuffer ring(dfb::ring);
    Noc noc;
    const auto acc_out = TensorAccessor(tensor::output);

    for (uint32_t i = get_my_thread_id(); i < num_pages; i += W) {
        noc.async_write<NocOptions::TXN_ID>(ring, acc_out, {}, {.page_id = start_page + i});
    }
    // Firmware waits for the tagged writes to land when the kernel returns; hand the tag back.
    noc_async_write_set_trid(0, noc.get_noc_id());
}
