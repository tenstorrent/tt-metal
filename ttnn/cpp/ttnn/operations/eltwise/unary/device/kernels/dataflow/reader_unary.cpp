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
    DataflowBuffer dfb_src(dfb::src);

#if SRC_SHARDED
    dfb_src.reserve_back(num_pages);
    dfb_src.push_back(num_pages);
#else
    constexpr uint32_t onepage = 1;
    const auto src = TensorAccessor(tensor::src);

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
            dfb_src.reserve_back(onepage);
            for (uint32_t r = 0; r < actual_rows; ++r) {
                noc.async_read(
                    src,
                    dfb_src,
                    bytes,
                    {.page_id = base_page + r, .offset_bytes = j * chunk_size},
                    {.offset_bytes = r * bytes});
            }
            noc.async_read_barrier();
            dfb_src.push_back(onepage);
        }
    }
#else
    const uint32_t page_bytes = dfb_src.get_entry_size();
#if DRAM_FLOW
    // Reads in the flow's page order, several pages per barrier. Bursts grow 1, 1, 2, 4, ... up to READ_BURST, so the
    // first tiles reach compute early.
    auto order = dram_read::RotatedPages::from_args();
    dram_read::DfbGroups groups{.depth = dfb_src.get_total_num_entries()};
    uint32_t burst = 1, next_burst = 1;
    auto max_burst = [&] {
        const uint32_t current = burst;
        burst = next_burst;
        next_burst = std::min(2 * next_burst, uint32_t{READ_BURST});
        return current;
    };
    auto read_group = [&](uint32_t n) {
        dfb_src.reserve_back(n);
        for (uint32_t k = 0; k < n; ++k) {
            noc.async_read(src, dfb_src, page_bytes, {.page_id = order.next()}, {.offset_bytes = k * page_bytes});
        }
        noc.async_read_barrier();
        dfb_src.push_back(n);
    };
    auto read_pages = [&](uint32_t first, uint32_t count) {
        dram_read::for_each_group(order, groups, first, count, max_burst, read_group);
    };
    dram_read::WorkSource::from_args(noc).for_each_chunk(read_pages);
#else
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_src.reserve_back(onepage);
        noc.async_read(src, dfb_src, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_src.push_back(onepage);
    }
#endif  // DRAM_FLOW
#endif
#endif
}
