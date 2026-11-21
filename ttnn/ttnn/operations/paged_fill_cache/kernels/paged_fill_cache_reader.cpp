// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// paged_fill_cache reader (op_design.md, Block schedule).
//
// Helper note: no kernel-lib helper moves a DRAM page into a dataflow buffer through a TensorAccessor
// (the design's "Helpers considered and rejected" table); this kernel uses the raw Noc / DataflowBuffer
// API, as Quasar's own data-movement ops do.
//
// Rings:
//   dfb_page_table_row  explicit sync (listed in disable_dfb_implicit_sync_for): the reader reserves both
//                       entries, reads row `batch_idx` of the page table into each (user trid 0, one
//                       barrier), pushes entry 0 to the writer and keeps entry 1 as its own lookup table,
//                       which it pushes as its last ring action.
//   dfb_input_tiles     implicit sync: one tagged read per mapped tile, no reserve/barrier/push; the DM0
//                       interrupt handler posts the credits per half-ring, firmware the last partial batch.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

#include "paged_fill_cache_walk.hpp"

void kernel_main() {
    const uint32_t core_start_tile = get_arg(args::core_start_tile);
    const uint32_t core_tiles = get_arg(args::core_tiles);
    const paged_fill_cache::WalkShape shape{
        .num_heads = get_arg(args::num_heads),
        .seq_tiles = get_arg(args::seq_tiles),
        .block_tiles = get_arg(args::block_tiles),
        .d_tiles = get_arg(args::d_tiles),
    };
    const uint32_t batch_idx = get_arg(args::batch_idx);
    const uint32_t pt_page_bytes = get_arg(args::pt_page_bytes);
    // The plan's count, not get_num_threads(): were a thread not to run, its tiles would be missing.
    constexpr uint32_t R = get_arg(args::reader_threads);
    // Copies of the row per reader thread (host PT_RING_ENTRIES): entry 0 for the writer, the rest the reader's.
    constexpr uint32_t ROW_COPIES = get_arg(args::page_table_row_copies);
    static_assert(ROW_COPIES >= 2, "one copy of the page-table row for each kernel");
    constexpr NocOptVals user_txn = {.trid = 0};

    DataflowBuffer dfb_input_tiles(dfb::input_tiles);
    DataflowBuffer dfb_page_table_row(dfb::page_table_row);
    Noc noc;
    const auto input_accessor = TensorAccessor(tensor::input);
    const auto page_table_accessor = TensorAccessor(tensor::page_table);
    const uint32_t pt_entry_bytes = dfb_page_table_row.get_entry_size();

    // load_page_table_row: ROW_COPIES copies of the row, one round trip.
    dfb_page_table_row.reserve_back(ROW_COPIES);
    for (uint32_t copy = 0; copy < ROW_COPIES; ++copy) {
        noc.async_read<NocOptions::TXN_ID>(
            page_table_accessor,
            dfb_page_table_row,
            pt_page_bytes,
            {.page_id = batch_idx},
            {.offset_bytes = copy * pt_entry_bytes},
            user_txn);
    }
    noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
    dfb_page_table_row.push_back(1);  // entry 0 -> the writer
    // Entry 1 stays reserved: producer-owned L1, the reader's lookup table for the whole stream.
    const auto* row = reinterpret_cast<const volatile tt_l1_ptr int32_t*>(dfb_page_table_row.get_write_ptr());

    // stream_mapped_tiles_in: this cluster's whole range is one block; unmapped tile rows are not read.
    paged_fill_cache::walk_mapped_tiles(
        row, shape, core_start_tile, core_tiles, get_my_thread_id(), R, [&](uint32_t input_page, uint32_t) {
            noc.async_read<NocOptions::TXN_ID>(input_accessor, dfb_input_tiles, {.page_id = input_page}, {});
        });

    // release_page_table_row: hand the reader's copy over; the writer pops every copy.
    dfb_page_table_row.push_back(ROW_COPIES - 1);

    // hand_back_noc_tags: the read tag register is sticky (target page, Ring protocols).
    noc_async_read_set_trid(0, noc.get_noc_id());
}
