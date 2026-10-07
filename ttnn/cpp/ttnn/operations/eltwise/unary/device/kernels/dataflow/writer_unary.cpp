// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#if WORK_QUEUE
#include "unary_work_queue.hpp"
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
#if WORK_QUEUE
    // DRAM height-sharded work queue (see unary_work_queue.hpp). The reader announces each chunk on
    // CB 5 as (first position, page count); count 0 ends. On the scheduler core this writer also runs
    // the scheduler: wherever it would wait on a circular buffer it answers requests instead, and after
    // its own chunks it keeps answering until every worker has been told to stop.
    constexpr uint32_t kWriteBurst = WRITE_BURST;
    const uint32_t depth = get_local_cb_interface(cb_id_dst).fifo_num_pages;
    const bool is_scheduler = get_arg_val<uint32_t>(1) != 0;
    const uint32_t total_pages = get_arg_val<uint32_t>(2);
    const uint32_t chunk_pages = get_arg_val<uint32_t>(3);
    const uint32_t num_workers = get_arg_val<uint32_t>(4);
    unary_wq::RotatedPages order{
        .shard_pages = get_arg_val<uint32_t>(5),
        .num_shards = get_arg_val<uint32_t>(6),
        .last_shard_pages = get_arg_val<uint32_t>(7)};
    unary_wq::Scheduler sched{
        .table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(unary_wq::kCbRequestTable)),
        .num_workers = num_workers,
        .total_chunks = unary_wq::num_chunks(total_pages, chunk_pages),
        .next_chunk = num_workers,  // chunks 0 .. num_workers - 1 are each worker's first, static chunk
        .coord_arg = dst_args.next_common_runtime_args_offset()};
    if (is_scheduler) {
        sched.start();
    }
    auto serve_until_available = [&](uint32_t cb_id, uint32_t n) {
        if (is_scheduler) {
            while (!cb_pages_available_at_front(cb_id, n)) {
                sched.serve();
            }
        }
    };
    uint32_t drained_in_cycle = 0;
    while (true) {
        serve_until_available(unary_wq::kCbWriterChunk, 1);
        cb_wait_front(unary_wq::kCbWriterChunk, 1);
        auto* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(unary_wq::kCbWriterChunk));
        const uint32_t first = w[0];
        const uint32_t count = w[1];
        cb_pop_front(unary_wq::kCbWriterChunk, 1);
        if (count == 0) {
            break;
        }
        order.seek(first);
        for (uint32_t done = 0; done < count;) {
            uint32_t n = (count - done < kWriteBurst) ? (count - done) : kWriteBurst;
            if (n > depth - drained_in_cycle) {
                n = depth - drained_in_cycle;
            }
            serve_until_available(cb_id_dst, n);
            dfb_dst.wait_front(n);
            for (uint32_t k = 0; k < n; ++k) {
                noc.async_write(dfb_dst, dst, page_bytes, {.offset_bytes = k * page_bytes}, {.page_id = order.next()});
            }
            noc.async_writes_flushed();
            dfb_dst.pop_front(n);
            drained_in_cycle += n;
            if (drained_in_cycle == depth) {
                drained_in_cycle = 0;
            }
            done += n;
        }
    }
    while (is_scheduler && !sched.finished()) {
        sched.serve();
    }
#elif SHARD_ROTATE
    // DRAM height-sharded: write back in the reader's order, slot by slot across the shards.
    // Same order and start (slot start_id, shard arg 6) as the reader.
    const uint32_t shard_pages = get_arg_val<uint32_t>(3);
    const uint32_t num_shards = get_arg_val<uint32_t>(4);
    const uint32_t last_shard_pages = get_arg_val<uint32_t>(5);
    uint32_t slot = start_id;
    uint32_t shard = get_arg_val<uint32_t>(6);
    auto next_page = [&]() -> uint32_t {
        if (shard == num_shards - 1 && slot >= last_shard_pages) {
            shard = 0;
            slot++;
        }
        const uint32_t page = shard * shard_pages + slot;
        if (++shard == num_shards) {
            shard = 0;
            slot++;
        }
        return page;
    };
    // Flush once per WRITE_BURST instead of once per tile, so a few writes share the round trip. The
    // output CB is two bursts deep so pack can fill the next group while these writes drain.
    constexpr uint32_t kWriteBurst = WRITE_BURST;
    const uint32_t depth = get_local_cb_interface(cb_id_dst).fifo_num_pages;
    uint32_t drained_in_cycle = 0;
    for (uint32_t done = 0; done < num_pages;) {
        uint32_t n = kWriteBurst;
        const uint32_t remaining = num_pages - done;
        const uint32_t until_wrap = depth - drained_in_cycle;
        if (n > remaining) {
            n = remaining;
        }
        if (n > until_wrap) {
            n = until_wrap;
        }
        dfb_dst.wait_front(n);
        for (uint32_t k = 0; k < n; ++k) {
            noc.async_write(dfb_dst, dst, page_bytes, {.offset_bytes = k * page_bytes}, {.page_id = next_page()});
        }
        noc.async_writes_flushed();
        dfb_dst.pop_front(n);
        drained_in_cycle += n;
        if (drained_in_cycle == depth) {
            drained_in_cycle = 0;
        }
        done += n;
    }
#else
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_dst.wait_front(onepage);
        noc.async_write(dfb_dst, dst, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        dfb_dst.pop_front(onepage);
    }
#endif  // WORK_QUEUE
    noc.async_write_barrier();
#endif
#endif
}
