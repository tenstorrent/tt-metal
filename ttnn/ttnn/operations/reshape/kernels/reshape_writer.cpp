// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// reshape writer (op_design.md, Block schedule). Implicit sync in both regimes: one tagged write per ring entry,
// the DM0 ISR acks the entries (no wait_front / barrier / pop_front); firmware waits for the writes at kernel exit.
// No kernel-library helper covers an interleaved page copy out of a DataflowBuffer (see reshape_reader.cpp).
//
// page_identity (default build): store_pages_block -- output page p = input page p over this cluster's range,
//   block by block. Thread t of W writes pages t, t+W, ... of the range (STRIDED ring dealing).
//
// tile_row_regroup (RESHAPE_TILE_ROW_REGROUP): store_tiles_block -- the output tiles in the order the relayout
//   pushes them: column block outer, row block inner, then output tile-row k, then column w; page k * Wt + w.
//   Thread t of W writes the entries t, t+W, ... of that sequence.
//
// The kernel never divides (Quasar XIP DM build).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    DataflowBuffer out_ring(dfb::out_ring);
    Noc noc;
    const auto acc_output = TensorAccessor(tensor::output);
    constexpr uint32_t W = get_arg(args::writer_threads);

#ifndef RESHAPE_TILE_ROW_REGROUP
    const uint32_t core_pages = get_arg(args::core_pages);
    const uint32_t start_page = get_arg(args::start_page);
    constexpr uint32_t block_pages = get_arg(args::block_pages);

    uint32_t next = get_my_thread_id();
    for (uint32_t block_start = 0; block_start < core_pages; block_start += block_pages) {
        const uint32_t block_end = block_start + block_pages < core_pages ? block_start + block_pages : core_pages;
        // store_pages_block(block_idx)
        for (; next < block_end; next += W) {
            noc.async_write<NocOptions::TXN_ID>(out_ring, acc_output, {}, {.page_id = start_page + next});
        }
    }
#else
    const uint32_t col_start = get_arg(args::col_start);
    const uint32_t col_count = get_arg(args::col_count);
    const uint32_t row_start = get_arg(args::row_start);
    const uint32_t row_count = get_arg(args::row_count);
    constexpr uint32_t tensor_w_tiles = get_arg(args::tensor_w_tiles);
    constexpr uint32_t block_tile_cols = get_arg(args::block_tile_cols);
    constexpr uint32_t block_out_tile_rows = get_arg(args::block_out_tile_rows);

    uint32_t entry = 0;                    // position in the relayout's push sequence
    uint32_t next = get_my_thread_id();    // the next entry this thread owns
    for (uint32_t cb = 0; cb < col_count; cb += block_tile_cols) {
        const uint32_t wb = cb + block_tile_cols < col_count ? block_tile_cols : col_count - cb;
        const uint32_t w0 = col_start + cb;
        for (uint32_t kb = 0; kb < row_count; kb += block_out_tile_rows) {
            const uint32_t nr = kb + block_out_tile_rows < row_count ? block_out_tile_rows : row_count - kb;
            // store_tiles_block(block_idx)
            uint32_t page_row = (row_start + kb) * tensor_w_tiles + w0;
            for (uint32_t k = 0; k < nr; ++k) {
                for (uint32_t w = 0; w < wb; ++w, ++entry) {
                    if (entry == next) {
                        noc.async_write<NocOptions::TXN_ID>(out_ring, acc_output, {}, {.page_id = page_row + w});
                        next += W;
                    }
                }
                page_row += tensor_w_tiles;
            }
        }
    }
#endif
    // Hand the hart back with the user id in its sticky tag register (target page, Ring protocols).
    noc_async_write_set_trid(0, noc.get_noc_id());
}
