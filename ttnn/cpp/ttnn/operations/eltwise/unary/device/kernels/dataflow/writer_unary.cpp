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
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto cb_id_dst = tt::CBIndex::c_2;

    Noc noc;
    DataflowBuffer dfb_dst(cb_id_dst);

#if DST_SHARDED
    // Output is sharded in place; the wait is only a readiness handshake. Pop to
    // leave the CB balanced.
    dfb_dst.wait_front(num_pages);
    dfb_dst.pop_front(num_pages);
#else
    constexpr uint32_t onepage = 1;
    constexpr auto dst_args = TensorAccessorArgs<0, 0>();
    const auto dst = TensorAccessor(dst_args, dst_addr);

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
    const uint32_t page_bytes = get_local_cb_interface(cb_id_dst).fifo_page_size;
#if SHARD_ROTATE
    // DRAM-sharded: the reader's rotated page order, WRITE_BURST pages per flush.
    constexpr uint32_t kWriteBurst = WRITE_BURST;
    auto order = dram_shard::RotatedPages::from_args(3);
    dram_shard::CbGroups groups{.depth = get_local_cb_interface(cb_id_dst).fifo_num_pages};
#if WORK_QUEUE
    // On the scheduler core, the writer answers requests whenever it would otherwise wait.
    const bool is_scheduler = get_arg_val<uint32_t>(1) != 0;
    dram_shard::Scheduler sched{
        .num_workers = get_arg_val<uint32_t>(9),
        .total_chunks = dram_shard::num_chunks(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(8)),
        .coord_arg = dst_args.next_common_runtime_args_offset()};
    if (is_scheduler) {
        sched.start();
    }
    auto serve_until = [&](uint32_t cb_id, uint32_t n) {
        while (is_scheduler && !cb_pages_available_at_front(cb_id, n)) {
            sched.serve();
        }
    };
#endif
    auto write_pages = [&](uint32_t count) {
        for (uint32_t done = 0; done < count;) {
            const uint32_t n = groups.next(count - done < kWriteBurst ? count - done : kWriteBurst);
#if WORK_QUEUE
            serve_until(cb_id_dst, n);
#endif
            dfb_dst.wait_front(n);
            for (uint32_t k = 0; k < n; ++k) {
                noc.async_write(dfb_dst, dst, page_bytes, {.offset_bytes = k * page_bytes}, {.page_id = order.next()});
            }
            noc.async_writes_flushed();
            dfb_dst.pop_front(n);
            done += n;
        }
    };
#if WORK_QUEUE
    while (true) {
        serve_until(dram_shard::kCbWriterChunk, 1);
        cb_wait_front(dram_shard::kCbWriterChunk, 1);
        auto* chunk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(dram_shard::kCbWriterChunk));
        const uint32_t first = chunk[0];
        const uint32_t count = chunk[1];
        cb_pop_front(dram_shard::kCbWriterChunk, 1);
        if (count == 0) {
            break;
        }
        order.seek(first);
        write_pages(count);
    }
    while (is_scheduler && !sched.finished()) {
        sched.serve();
    }
#else
    order.seek(start_id);
    write_pages(num_pages);
#endif
#else
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_dst.wait_front(onepage);
        noc.async_write(dfb_dst, dst, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        dfb_dst.pop_front(onepage);
    }
#endif  // SHARD_ROTATE
    noc.async_write_barrier();
#endif
#endif
}
