// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// paged_fill_cache writer (op_design.md, Block schedule).
//
// Helper note: no kernel-lib helper scatters dataflow-buffer entries to TensorAccessor pages; this kernel
// uses the raw Noc / DataflowBuffer API (the design's "Helpers considered and rejected" table).
//
// Rings:
//   dfb_page_table_row  explicit sync: entry 0 (the row, pushed by the reader before its first tile read)
//                       is waited for once and held for the whole stream; entry 1 (the reader's own copy,
//                       pushed at the reader's end) is waited for and popped after the last write.
//   dfb_input_tiles     implicit sync: one tagged write per tile, no wait/barrier/pop; the DM0 interrupt
//                       handler acks the entries, firmware waits for the writes to land at kernel exit.
//
// The writer writes only into the cache tensor's own pages: the op is in place (rule R3) and never reads
// the cache.

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
    constexpr uint32_t W = get_arg(args::writer_threads);
    // Copies of the row per thread (host PT_RING_ENTRIES): entry 0 is this kernel's, the rest the reader's.
    constexpr uint32_t ROW_COPIES = get_arg(args::page_table_row_copies);

    DataflowBuffer dfb_input_tiles(dfb::input_tiles);
    DataflowBuffer dfb_page_table_row(dfb::page_table_row);
    Noc noc;
    const auto cache_accessor = TensorAccessor(tensor::cache);

    dfb_page_table_row.wait_front(1);
    const auto* row = reinterpret_cast<const volatile tt_l1_ptr int32_t*>(dfb_page_table_row.get_read_ptr());

    // scatter_tiles_to_cache: the same walk and skip as the reader; each tile to its cache page.
    paged_fill_cache::walk_mapped_tiles(
        row, shape, core_start_tile, core_tiles, get_my_thread_id(), W, [&](uint32_t, uint32_t cache_page) {
            noc.async_write<NocOptions::TXN_ID>(dfb_input_tiles, cache_accessor, {}, {.page_id = cache_page});
        });

    // release_page_table_row: every copy, ROW_COPIES pushed / ROW_COPIES popped.
    dfb_page_table_row.pop_front(1);
    dfb_page_table_row.wait_front(ROW_COPIES - 1);
    dfb_page_table_row.pop_front(ROW_COPIES - 1);

    // hand_back_noc_tags (see the reader).
    noc_async_write_set_trid(0, noc.get_noc_id());
}
