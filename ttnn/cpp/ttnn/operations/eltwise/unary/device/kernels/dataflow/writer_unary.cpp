// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#if DRAM_FLOW
#include "dram_read.hpp"
#endif

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb_dst(dfb::dst);

#if DST_SHARDED
    // Output is sharded in place; the wait is only a readiness handshake. Pop to
    // leave the DFB balanced.
    dfb_dst.wait_front(num_pages);
    dfb_dst.pop_front(num_pages);
#else
    constexpr uint32_t onepage = 1;
    const auto dst = TensorAccessor(tensor::dst);

    uint32_t end_id = start_id + num_pages;
#if RM_INTERLEAVED
    const uint32_t chunks_per_row = get_arg(args::chunks_per_row);
    const uint32_t chunk_size = get_arg(args::chunk_size);
    const uint32_t last_chunk_size = get_arg(args::last_chunk_size);
    const uint32_t rows_per_tile = get_arg(args::rows_per_tile);
    const uint32_t total_rows = get_arg(args::total_rows);

    for (uint32_t block = start_id; block < end_id; ++block) {
        uint32_t base_page = block * rows_per_tile;
        uint32_t remaining = total_rows - base_page;
        uint32_t actual_rows = (rows_per_tile < remaining) ? rows_per_tile : remaining;

        for (uint32_t j = 0; j < chunks_per_row; ++j) {
            uint32_t bytes = (j == chunks_per_row - 1) ? last_chunk_size : chunk_size;
            dfb_dst.wait_front(onepage);
            for (uint32_t r = 0; r < actual_rows; ++r) {
                noc.async_write(
                    dfb_dst,
                    dst,
                    bytes,
                    {.offset_bytes = r * bytes},
                    {.page_id = base_page + r, .offset_bytes = j * chunk_size});
            }
            noc.async_writes_flushed();
            dfb_dst.pop_front(onepage);
        }
    }
    noc.async_write_barrier();
#else
    const uint32_t page_bytes = dfb_dst.get_entry_size();
#if DRAM_FLOW
    // Writes in the reader's page order, WRITE_BURST pages per flush.
    auto order = dram_read::RotatedPages::from_args();
    dram_read::DfbGroups groups{.depth = dfb_dst.get_total_num_entries()};
    auto work = dram_read::WorkSource::from_args(noc);
    auto write_group = [&](uint32_t n) {
        // The scheduler core answers requests while its own pages are on the way.
        work.serve_until([&] { return dfb_dst.pages_available_at_front(n); });
        dfb_dst.wait_front(n);
        for (uint32_t k = 0; k < n; ++k) {
            noc.async_write(dfb_dst, dst, page_bytes, {.offset_bytes = k * page_bytes}, {.page_id = order.next()});
        }
        noc.async_writes_flushed();
        dfb_dst.pop_front(n);
    };
    auto write_pages = [&](uint32_t first, uint32_t count) {
        dram_read::for_each_group(order, groups, first, count, [] { return uint32_t{WRITE_BURST}; }, write_group);
    };
    work.for_each_chunk(write_pages);
#else
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_dst.wait_front(onepage);
        noc.async_write(dfb_dst, dst, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        dfb_dst.pop_front(onepage);
    }
#endif  // DRAM_FLOW
    noc.async_write_barrier();
#endif
#endif
}
