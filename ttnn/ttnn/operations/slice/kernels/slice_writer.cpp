// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// slice writer (store_block): drains the `slice_pages` ring into this cluster's contiguous range of output
// pages.
//
// Dealing: hart w of W writes output pages w, w+W, ... of [first_out_page, first_out_page + num_pages),
// the same entries the matching reader hart filled (STRIDED ring on both ends).
//
// Ring protocol (CT arg explicit_sync, see slice_reader.cpp):
//   implicit: a tagged write leaves from the ring's cursor and the DM0 interrupt handler acks one block per
//     batch: no wait/barrier/pop; firmware waits for the writes to land at kernel exit.
//   explicit: wait for a block, issue its writes (user trid 0), one barrier, pop the block.

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
    const uint32_t thread_block_pages = get_arg(args::thread_block_pages);
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr bool explicit_sync = get_arg(args::explicit_sync) != 0;

    DataflowBuffer cb_slice_pages(dfb::slice_pages);
    Noc noc;
    const auto output_accessor = TensorAccessor(tensor::output);
    const uint32_t thread = get_my_thread_id();

    if constexpr (!explicit_sync) {
        for (uint32_t i = thread; i < num_pages; i += W) {
            noc.async_write<NocOptions::TXN_ID>(cb_slice_pages, output_accessor, {}, {.page_id = first_out_page + i});
        }
    } else if (thread < num_pages) {
        constexpr NocOptVals user_txn = {.trid = 0};
        const uint32_t entry_bytes = cb_slice_pages.get_entry_size();
        const uint32_t entry_stride_bytes = W * entry_bytes;
        uint32_t page = first_out_page + thread;
        uint32_t remaining = (num_pages - thread + W - 1) / W;  // this hart's pages
        while (remaining > 0) {
            const uint32_t block = remaining < thread_block_pages ? remaining : thread_block_pages;
            cb_slice_pages.wait_front(block);
            for (uint32_t j = 0; j < block; ++j, page += W) {
                noc.async_write<NocOptions::TXN_ID>(
                    cb_slice_pages,
                    output_accessor,
                    entry_bytes,
                    {.offset_bytes = j * entry_stride_bytes},
                    {.page_id = page},
                    user_txn);
            }
            noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
            cb_slice_pages.pop_front(block);
            remaining -= block;
        }
    }
    // Hand the hart's sticky tag register back with the user id (see slice_reader.cpp).
    noc_async_write_set_trid(0, noc.get_noc_id());
}
