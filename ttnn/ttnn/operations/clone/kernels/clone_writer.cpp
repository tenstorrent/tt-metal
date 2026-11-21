// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// clone writer (block operation `store_block`, op_design.md): the ring `staged_pages` out to this
// cluster's contiguous page range [start_page, start_page + core_pages) of the output.
//
// Raw dataflow API, no kernel-lib helper (see clone_reader.cpp). Implicit sync: one tagged write per
// entry leaves from the ring's cursor and the DM0 interrupt handler acks it, so there is no wait /
// barrier / pop. Hart t of W writes entries t, t+W, ..., which are pages start_page + t, + t + W, ...

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t core_pages = get_arg(args::core_pages);
    const uint32_t start_page = get_arg(args::start_page);
    constexpr uint32_t W = get_arg(args::writer_threads);

    DataflowBuffer cb_staged_pages(dfb::staged_pages);
    Noc noc;
    const auto acc_output = TensorAccessor(tensor::output);

    for (uint32_t i = get_my_thread_id(); i < core_pages; i += W) {
        noc.async_write<NocOptions::TXN_ID>(cb_staged_pages, acc_output, {}, {.page_id = start_page + i});
    }
    // Firmware waits for the tagged writes to land when the kernel returns. Hand the tag back.
    noc_async_write_set_trid(0, noc.get_noc_id());
}
