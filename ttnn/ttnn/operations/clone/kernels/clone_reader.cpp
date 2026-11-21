// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// clone reader (block operation `load_block`, op_design.md): this cluster's contiguous page range
// [start_page, start_page + core_pages) of the input into the STRIDED ring `staged_pages`.
//
// Raw dataflow API, no kernel-lib helper: no helper covers an interleaved page copy (op_design.md,
// "Helpers considered and rejected"). Implicit sync: one tagged read per page lands at the ring's cursor
// and the DM0 interrupt handler posts the credit, so there is no reserve / barrier / push. The block is
// the whole range, streamed through the ring window; hart t of R reads pages t, t+R, ...

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
    constexpr uint32_t R = get_arg(args::reader_threads);

    DataflowBuffer cb_staged_pages(dfb::staged_pages);
    Noc noc;
    const auto acc_input = TensorAccessor(tensor::input);

    for (uint32_t i = get_my_thread_id(); i < core_pages; i += R) {
        noc.async_read<NocOptions::TXN_ID>(acc_input, cb_staged_pages, {.page_id = start_page + i}, {});
    }
    // Firmware credits a last partial batch when the kernel returns. Hand the hart's sticky transaction
    // tag back.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
