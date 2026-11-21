// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// to_memory_config reader: load_block of op_design.md. The entries of this core's range (entry id
// first_entry + (i / run_len) * run_stride + (i % run_len) * entry_stride) are read through the input's
// TensorAccessor (a DRAM bank, an L1 bank, or the core holding the page's shard) into the staging ring
// `dfb_pages`, block_pages entries per block.
//
// An entry is in_pieces input pages: entry g holds pages g * in_pieces + p (op_design.md, "Entry space"). A tile,
// or a row-major row/run that is one input page, is one piece: one implicit-sync read per entry, tagged
// NocOptions::TXN_ID, landing at the ring's cursor; DM0's interrupt handler posts its credit. Several pieces (a
// row-major row assembled from width-shard pages) are one entry assembled by several sized reads, so this side keeps
// the credits itself (explicit sync, the ring opted out on the host) and tags every read with the user id. Quasar's
// NoC needs no address alignment, so a piece lands at any byte offset of the entry.
// Thread plan: reader hart t of R takes entries t, t+R, ... of the core's range.

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
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t block_pages = get_arg(args::block_pages);
    constexpr uint32_t pieces = get_arg(args::in_pieces);
    constexpr uint32_t piece_bytes = get_arg(args::in_piece_bytes);
    constexpr NocOptVals user_txn = {.trid = 0};

    DataflowBuffer dfb_pages(dfb::dfb_pages);
    Noc noc;
    const auto input = TensorAccessor(tensor::input);

    uint32_t i = get_my_thread_id();  // this hart's next entry index in the core's range
    for (uint32_t block_start = 0; block_start < entry_count; block_start += block_pages) {
        // load_block: the last block is short (entry_count - block_start entries); same operation.
        const uint32_t block_end =
            (entry_count - block_start < block_pages) ? entry_count : block_start + block_pages;
        for (; i < block_end; i += R) {
            const uint32_t entry = first_entry + (i / run_len) * run_stride + (i % run_len) * entry_stride;
            if constexpr (pieces == 1) {
                noc.async_read<NocOptions::TXN_ID>(input, dfb_pages, {.page_id = entry}, {});
            } else {
                dfb_pages.reserve_back(1);
                for (uint32_t p = 0; p < pieces; ++p) {
                    noc.async_read<NocOptions::TXN_ID>(
                        input, dfb_pages, piece_bytes, {.page_id = entry * pieces + p},
                        {.offset_bytes = p * piece_bytes}, user_txn);
                }
                noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
                dfb_pages.push_back(1);
            }
        }
    }
    if constexpr (pieces == 1) {
        // No barrier: firmware credits a last partial batch and drains the ring when the kernel returns.
        // Hand the hart back with the user transaction id in its sticky tag register (target page, "Ring protocols").
        // The explicit form already ends on a user-id read.
        noc_async_read_set_trid(0, noc.get_noc_id());
    }
}
