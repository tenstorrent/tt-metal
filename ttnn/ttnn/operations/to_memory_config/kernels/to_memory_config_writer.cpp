// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// to_memory_config writer: store_block of op_design.md. The entries of the staging ring `dfb_pages` go out
// through the output's TensorAccessor under the entry ids the reader read them from, block_pages entries per block.
//
// An entry is out_pieces output pages: entry g holds pages g * out_pieces + p (op_design.md, "Entry space"). One
// piece: one implicit-sync write per entry, tagged NocOptions::TXN_ID, leaving from the ring's cursor; DM0's interrupt
// handler acks the entry once it is sent and firmware waits for the writes to land when the kernel returns. Several
// pieces (a row-major row split into width-shard pages): this side keeps the credits itself (explicit sync, the ring
// opted out on the host), tags every sized write with the user id and barriers before it frees the entry.
// Thread plan: writer hart t of W takes entries t, t+W, ... of the core's range.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t first_entry = get_arg(args::first_entry);
    const uint32_t entry_count = get_arg(args::entry_count);
    const uint32_t entry_stride = get_arg(args::entry_stride);
    const uint32_t run_len = get_arg(args::run_len);
    const uint32_t run_stride = get_arg(args::run_stride);
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t block_pages = get_arg(args::block_pages);
    constexpr uint32_t pieces = get_arg(args::out_pieces);
    constexpr uint32_t piece_bytes = get_arg(args::out_piece_bytes);
    constexpr NocOptVals user_txn = {.trid = 0};

    DataflowBuffer dfb_pages(dfb::dfb_pages);
    Noc noc;
    const auto output = TensorAccessor(tensor::output);

    uint32_t i = get_my_thread_id();  // this hart's next entry index in the core's range
    for (uint32_t block_start = 0; block_start < entry_count; block_start += block_pages) {
        // store_block: the last block is short (entry_count - block_start entries); same operation.
        const uint32_t block_end =
            (entry_count - block_start < block_pages) ? entry_count : block_start + block_pages;
        for (; i < block_end; i += W) {
            const uint32_t entry = first_entry + (i / run_len) * run_stride + (i % run_len) * entry_stride;
            if constexpr (pieces == 1) {
                noc.async_write<NocOptions::TXN_ID>(dfb_pages, output, {}, {.page_id = entry});
            } else {
                dfb_pages.wait_front(1);
                for (uint32_t p = 0; p < pieces; ++p) {
                    noc.async_write<NocOptions::TXN_ID>(
                        dfb_pages, output, piece_bytes, {.offset_bytes = p * piece_bytes},
                        {.page_id = entry * pieces + p}, user_txn);
                }
                noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
                dfb_pages.pop_front(1);
            }
        }
    }
    if constexpr (pieces == 1) {
        // Hand the hart back with the user transaction id in its sticky tag register (see the reader).
        noc_async_write_set_trid(0, noc.get_noc_id());
    }
}
