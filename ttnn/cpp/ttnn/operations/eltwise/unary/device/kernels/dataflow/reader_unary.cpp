// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#if SHARD_ROTATE
#include "dram_height_sharded.hpp"
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
    // DRAM height-sharded: rotated page order (dram_height_sharded.hpp), READ_BURST pages per barrier. Bursts ramp
    // 1, 1, 2, 4, ... so the first tile reaches compute without waiting for a full burst.
    constexpr uint32_t kReadBurst = READ_BURST;
    dram_hs::RotatedPages order{
        .shard_pages = get_arg_val<uint32_t>(3),
        .num_shards = get_arg_val<uint32_t>(4),
        .last_shard_pages = get_arg_val<uint32_t>(5)};
    dram_hs::CbGroups groups{.depth = get_local_cb_interface(cb_id_src).fifo_num_pages};
    uint32_t burst = 1, next_burst = 1;
    auto read_pages = [&](uint32_t count) {
        for (uint32_t done = 0; done < count;) {
            const uint32_t n = groups.next(count - done < burst ? count - done : burst);
            dfb_src.reserve_back(n);
            for (uint32_t k = 0; k < n; ++k) {
                noc.async_read(src, dfb_src, page_bytes, {.page_id = order.next()}, {.offset_bytes = k * page_bytes});
            }
            noc.async_read_barrier();
            dfb_src.push_back(n);
            done += n;
            burst = next_burst;
            next_burst = 2 * next_burst < kReadBurst ? 2 * next_burst : kReadBurst;
        }
    };
#if WORK_QUEUE
    // Start on chunk worker_id; ask for the next chunk while reading the current one.
    const uint32_t worker_id = get_arg_val<uint32_t>(1);
    const uint32_t total_pages = get_arg_val<uint32_t>(2);
    const uint32_t chunk_pages = get_arg_val<uint32_t>(6);
    const uint32_t total_chunks = dram_hs::num_chunks(total_pages, chunk_pages);
    dram_hs::Client client{
        .request_addr =
            dram_hs::noc_addr(get_arg_val<uint32_t>(7), get_write_ptr(dram_hs::kCbRequestTable) + 4 * worker_id)};
    for (uint32_t chunk = worker_id; chunk != dram_hs::kDone; chunk = client.receive()) {
        const bool asked = client.try_request();  // until the scheduler starts, ask after reading instead
        if (chunk < total_chunks) {
            const uint32_t first = chunk * chunk_pages;
            const uint32_t count = total_pages - first < chunk_pages ? total_pages - first : chunk_pages;
            dram_hs::announce(first, count);
            order.seek(first);
            read_pages(count);
        }
        if (!asked) {
            client.request();
        }
    }
    noc_async_write_barrier();  // flush the inline request writes
    dram_hs::announce(0, 0);
#else
    order.seek(start_id);
    read_pages(num_pages);
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
