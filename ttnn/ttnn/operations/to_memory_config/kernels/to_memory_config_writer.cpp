// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// to_memory_config writer: store_block of op_design.md. The entries of the staging ring `dfb_pages` go out
// through the output's TensorAccessor under the same page ids the reader read them from, block_pages pages
// per block.
//
// Implicit sync: each write is tagged NocOptions::TXN_ID, leaves from the ring's cursor, and DM0's interrupt
// handler acks the entry once it is sent: no wait / barrier / pop here. Firmware waits for the writes to land
// when the kernel returns.
// Thread plan: writer hart t of W takes pages t, t+W, ... of the core's range.

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
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t block_pages = get_arg(args::block_pages);

    DataflowBuffer dfb_pages(dfb::dfb_pages);
    Noc noc;
    const auto output = TensorAccessor(tensor::output);

    uint32_t i = get_my_thread_id();  // this hart's next page index in the core's range
    for (uint32_t block_start = 0; block_start < num_pages; block_start += block_pages) {
        // store_block: the last block is short (num_pages - block_start pages); same operation.
        const uint32_t block_end = (num_pages - block_start < block_pages) ? num_pages : block_start + block_pages;
        for (; i < block_end; i += W) {
            noc.async_write<NocOptions::TXN_ID>(dfb_pages, output, {}, {.page_id = start_page + i * page_stride});
        }
    }
    // Hand the hart back with the user transaction id in its sticky tag register (see the reader).
    noc_async_write_set_trid(0, noc.get_noc_id());
}
