// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// reshape reader (op_design.md, Block schedule).
//
// No kernel-library helper covers an interleaved page copy into a DataflowBuffer (the library's dataflow helpers
// are scalers, mcast, tile columns and local copies); the NoC calls below are the target's prescribed pattern
// (.claude/references/quasar_op_examples/toy_thread_plan/kernels/reader*.cpp).
//
// page_identity (default build): load_pages_block -- this cluster's pages [start_page, start_page + core_pages)
//   into `in_ring`, block by block, one implicit-sync read per page (the DM0 ISR posts the credits, one txn-id
//   batch per block). Thread t of R reads pages t, t+R, ... of the cluster's range (STRIDED ring dealing).
//
// tile_row_regroup and w_regroup_rm (RESHAPE_WINDOW_READER): load_window_block -- for each block (column block
//   outer, row block inner) the input tile-rows [window_first, window_first + window_rows) x the block's tile
//   columns, read tile-row-major into window entry (tr - window_first) * block_tile_cols + (w - w0). Regime
//   w_regroup_rm is the one-column case: tensor_w_tiles = block_tile_cols = 1, a "tile-row" is one row-major input
//   page, and page_bytes is the window entry (the page rounded up to the L1 alignment, inside the page's slot). Explicit sync: one
//   reserve_back / barrier / push_back of the nominal `window_entries` per block, so the window never wraps; every
//   transfer carries the user transaction id.
//
// The kernel never divides (Quasar XIP DM build): every cursor is an increment or a multiply-add.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    DataflowBuffer in_ring(dfb::in_ring);
    Noc noc;
    const auto acc_input = TensorAccessor(tensor::input);

#ifndef RESHAPE_WINDOW_READER
    const uint32_t core_pages = get_arg(args::core_pages);
    const uint32_t start_page = get_arg(args::start_page);
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t block_pages = get_arg(args::block_pages);

    uint32_t next = get_my_thread_id();  // the next page index (within this cluster's range) this thread owns
    for (uint32_t block_start = 0; block_start < core_pages; block_start += block_pages) {
        const uint32_t block_end = block_start + block_pages < core_pages ? block_start + block_pages : core_pages;
        // load_pages_block(block_idx)
        for (; next < block_end; next += R) {
            noc.async_read<NocOptions::TXN_ID>(acc_input, in_ring, {.page_id = start_page + next}, {});
        }
    }
    // Firmware credits a last partial batch at kernel exit. Hand the hart back with the user id in its sticky tag
    // register.
    noc_async_read_set_trid(0, noc.get_noc_id());
#else
    const uint32_t col_start = get_arg(args::col_start);
    const uint32_t col_count = get_arg(args::col_count);
    const uint32_t num_row_blocks = get_arg(args::num_row_blocks);
    constexpr uint32_t tensor_w_tiles = get_arg(args::tensor_w_tiles);
    constexpr uint32_t block_tile_cols = get_arg(args::block_tile_cols);
    constexpr uint32_t window_entries = get_arg(args::window_entries);
    constexpr uint32_t page_bytes = get_arg(args::page_bytes);
    constexpr NocOptVals user_txn = {.trid = 0};
    static_assert(get_arg(args::reader_threads) == 1, "the window is assembled by one reader thread");

    for (uint32_t cb = 0; cb < col_count; cb += block_tile_cols) {
        const uint32_t wb = cb + block_tile_cols < col_count ? block_tile_cols : col_count - cb;
        const uint32_t w0 = col_start + cb;
        for (uint32_t rb = 0; rb < num_row_blocks; ++rb) {
            // load_window_block(block_idx)
            const uint32_t window_first = get_arg(args::window_first)[rb];
            const uint32_t window_rows = get_arg(args::window_rows)[rb];
            in_ring.reserve_back(window_entries);
            uint32_t page_row = window_first * tensor_w_tiles + w0;  // page id of (tile-row, column w0)
            uint32_t entry_row = 0;                                  // window entry of (tile-row, column 0)
            for (uint32_t i = 0; i < window_rows; ++i) {
                for (uint32_t w = 0; w < wb; ++w) {
                    noc.async_read<NocOptions::TXN_ID>(
                        acc_input,
                        in_ring,
                        page_bytes,
                        {.page_id = page_row + w},
                        {.offset_bytes = (entry_row + w) * page_bytes},
                        user_txn);
                }
                page_row += tensor_w_tiles;
                entry_row += block_tile_cols;
            }
            noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
            in_ring.push_back(window_entries);
        }
    }
#endif
}
