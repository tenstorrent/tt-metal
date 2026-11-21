// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// slice writer (store_block): drains the `slice_pages` ring into this cluster's contiguous range of output
// pages. Implicit sync: a write tagged NocOptions::TXN_ID leaves from the ring's cursor and the DM0 interrupt
// handler acks one block per batch; no wait / barrier / pop here.
//
// Dealing: hart w of W writes output pages w, w+W, ... of [first_out_page, first_out_page + num_pages),
// the same entries the matching reader hart filled (STRIDED ring on both ends).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t first_out_page = get_arg(args::first_out_page);
    const uint32_t num_pages = get_arg(args::num_pages);
    constexpr uint32_t W = get_arg(args::writer_threads);

    DataflowBuffer cb_slice_pages(dfb::slice_pages);
    Noc noc;
    const auto output_accessor = TensorAccessor(tensor::output);

    for (uint32_t i = get_my_thread_id(); i < num_pages; i += W) {
        noc.async_write<NocOptions::TXN_ID>(cb_slice_pages, output_accessor, {}, {.page_id = first_out_page + i});
    }
    // Firmware waits for the tagged writes to land when the kernel returns.
    // Hand the hart's sticky tag register back with the user id (see slice_reader.cpp).
    noc_async_write_set_trid(0, noc.get_noc_id());
}
