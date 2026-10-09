// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#if SHARD_ROTATE
#include "dram_sharded.hpp"
#endif

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto cb_id_src = tt::CBIndex::c_0;

    Noc noc;
    DataflowBuffer dfb_src(cb_id_src);

#if SRC_SHARDED
    dfb_src.reserve_back(num_pages);
    dfb_src.push_back(num_pages);
#else
    constexpr uint32_t onepage = 1;
    constexpr auto src_args = TensorAccessorArgs<0, 0>();
    const auto src = TensorAccessor(src_args, src_addr);

    uint32_t end_id = start_id + num_pages;
#if RM_INTERLEAVED
    const uint32_t chunks_per_row = get_arg_val<uint32_t>(3);
    const uint32_t chunk_size = get_arg_val<uint32_t>(4);
    const uint32_t last_chunk_size = get_arg_val<uint32_t>(5);
    const uint32_t rows_per_tile = get_arg_val<uint32_t>(6);
    const uint32_t total_rows = get_arg_val<uint32_t>(7);

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
    const uint32_t page_bytes = get_local_cb_interface(cb_id_src).fifo_page_size;
#if SHARD_ROTATE
    // DRAM-sharded: rotated page order, READ_BURST pages per barrier. Bursts ramp 1, 1, 2, 4, ... so compute starts
    // without waiting for a full burst.
    auto order = dram_shard::RotatedPages::from_args();
    dram_shard::CbGroups groups{.depth = get_local_cb_interface(cb_id_src).fifo_num_pages};
    uint32_t burst = 1, next_burst = 1;
    auto read_chunk = [&](uint32_t first, uint32_t count) {
        order.seek(first);
        for (uint32_t done = 0; done < count;) {
            const uint32_t n = groups.next(std::min(count - done, burst));
            dfb_src.reserve_back(n);
            for (uint32_t k = 0; k < n; ++k) {
                noc.async_read(src, dfb_src, page_bytes, {.page_id = order.next()}, {.offset_bytes = k * page_bytes});
            }
            noc.async_read_barrier();
            dfb_src.push_back(n);
            done += n;
            burst = next_burst;
            next_burst = std::min(2 * next_burst, uint32_t{READ_BURST});
        }
    };
#if WORK_QUEUE
    dram_shard::QueueReader::from_args().for_each_chunk(read_chunk);
#else
    read_chunk(start_id, num_pages);
#endif
#else
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_src.reserve_back(onepage);
        noc.async_read(src, dfb_src, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_src.push_back(onepage);
    }
#endif  // SHARD_ROTATE
#endif
#endif
}
