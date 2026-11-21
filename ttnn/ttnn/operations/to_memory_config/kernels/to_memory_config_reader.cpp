// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// to_memory_config reader: load_block of op_design.md. Pages [start_page + i * page_stride] of this core's
// range are read through the input's TensorAccessor (a DRAM bank, an L1 bank, or the core holding the page's
// shard) into the staging ring `dfb_pages`, block_pages pages per block.
//
// Implicit sync: each read is tagged NocOptions::TXN_ID, lands at the ring's cursor, and DM0's interrupt
// handler posts its credit, so there is no reserve / barrier / push here. One NoC read per page is what the
// data needs (interleaved pages sit in different banks; an implicit STRIDED transfer is one entry).
// Thread plan: reader hart t of R takes pages t, t+R, ... of the core's range.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_page = get_arg(args::start_page);
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t page_stride = get_arg(args::page_stride);
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t block_pages = get_arg(args::block_pages);

    DataflowBuffer dfb_pages(dfb::dfb_pages);
    Noc noc;
    const auto input = TensorAccessor(tensor::input);

    uint32_t i = get_my_thread_id();  // this hart's next page index in the core's range
    for (uint32_t block_start = 0; block_start < num_pages; block_start += block_pages) {
        // load_block: the last block is short (num_pages - block_start pages); same operation.
        const uint32_t block_end = (num_pages - block_start < block_pages) ? num_pages : block_start + block_pages;
        for (; i < block_end; i += R) {
            noc.async_read<NocOptions::TXN_ID>(input, dfb_pages, {.page_id = start_page + i * page_stride}, {});
        }
    }
    // No barrier: firmware credits a last partial batch and drains the ring when the kernel returns.
    // Hand the hart back with the user transaction id in its sticky tag register (target page, "Ring protocols").
    noc_async_read_set_trid(0, noc.get_noc_id());
}
