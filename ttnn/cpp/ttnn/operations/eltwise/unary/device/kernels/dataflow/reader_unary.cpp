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
#if WORK_QUEUE
    // DRAM height-sharded work queue (see unary_work_queue.hpp). This core starts on chunk worker_id,
    // then asks the scheduler for the next chunk while it reads the current one. Each chunk is read in
    // the rotated order with full bursts (no ramp: the next chunk number is already on its way).
    constexpr uint32_t kReadBurst = READ_BURST;
    const uint32_t depth = get_local_cb_interface(cb_id_src).fifo_num_pages;
    const uint32_t worker_id = get_arg_val<uint32_t>(1);
    const uint32_t sched_x = get_arg_val<uint32_t>(2);
    const uint32_t sched_y = get_arg_val<uint32_t>(3);
    const uint32_t total_pages = get_arg_val<uint32_t>(4);
    const uint32_t chunk_pages = get_arg_val<uint32_t>(5);
    unary_wq::RotatedPages order{
        .shard_pages = get_arg_val<uint32_t>(6),
        .num_shards = get_arg_val<uint32_t>(7),
        .last_shard_pages = get_arg_val<uint32_t>(8)};
    const uint32_t total_chunks = unary_wq::num_chunks(total_pages, chunk_pages);
    unary_wq::Client client{
        .request_noc_addr = get_noc_addr(sched_x, sched_y, get_write_ptr(unary_wq::kCbRequestTable) + 4 * worker_id),
        .reply = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(unary_wq::kReplySemaphore)),
        .go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(unary_wq::kGoSemaphore))};
    auto announce = [&](uint32_t first, uint32_t count) {
        cb_reserve_back(unary_wq::kCbComputeCount, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(unary_wq::kCbComputeCount))[0] = count;
        cb_push_back(unary_wq::kCbComputeCount, 1);
        cb_reserve_back(unary_wq::kCbWriterChunk, 1);
        auto* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(unary_wq::kCbWriterChunk));
        w[0] = first;
        w[1] = count;
        cb_push_back(unary_wq::kCbWriterChunk, 1);
    };
    uint32_t filled_in_cycle = 0;
    uint32_t chunk = worker_id;
    while (true) {
        // Ask for the next chunk before reading this one. On the first chunk the scheduler may not
        // have raised the go flag yet; then read first and ask afterwards.
        const bool asked = client.go_raised();
        if (asked) {
            client.request();
        }
        if (chunk < total_chunks) {
            const uint32_t first = chunk * chunk_pages;
            const uint32_t count = (total_pages - first < chunk_pages) ? (total_pages - first) : chunk_pages;
            announce(first, count);
            order.seek(first);
            for (uint32_t done = 0; done < count;) {
                uint32_t n = (count - done < kReadBurst) ? (count - done) : kReadBurst;
                if (n > depth - filled_in_cycle) {
                    n = depth - filled_in_cycle;
                }
                dfb_src.reserve_back(n);
                for (uint32_t k = 0; k < n; ++k) {
                    noc.async_read(
                        src, dfb_src, page_bytes, {.page_id = order.next()}, {.offset_bytes = k * page_bytes});
                }
                noc.async_read_barrier();
                dfb_src.push_back(n);
                filled_in_cycle += n;
                if (filled_in_cycle == depth) {
                    filled_in_cycle = 0;
                }
                done += n;
            }
        }
        if (!asked) {
            client.request();
        }
        chunk = client.receive();
        if (chunk == unary_wq::kDone) {
            break;
        }
    }
    noc_async_write_barrier();  // the inline request writes
    announce(0, 0);
#elif SHARD_ROTATE
    // DRAM height-sharded. A shard's pages sit in one bank, so reading start_id, start_id + 1, ...
    // keeps every read of a burst on one bank. Pages are instead visited slot by slot across the
    // shards (slot 0 of shard 0, 1, ..., then slot 1, ...), so consecutive reads, and each burst,
    // spread over the banks the way interleaved pages do. This core takes num_pages of that order
    // from slot start_id, shard arg 6. A short last shard has no page at slots >= last_shard_pages.
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
    // Post READ_BURST pages, then one barrier. The burst ramps 1, 1, 2, 4, ... up to READ_BURST so the
    // first page reaches compute at one-page latency instead of waiting behind a full first burst from
    // every core. For a power-of-two burst the ramp sums to one burst, so the first full burst starts on
    // a CB-depth boundary. The CB is two bursts deep, so the next burst can be posted while compute
    // consumes the previous one. A burst never crosses the CB wrap: offsets are contiguous from the
    // write pointer.
    constexpr uint32_t kReadBurst = READ_BURST;
    uint32_t ramp = 1;
    uint32_t first_burst = 1;
    auto next_burst = [&]() -> uint32_t {
        const uint32_t n = ramp;
        if (first_burst) {
            first_burst = 0;
        } else if (ramp < kReadBurst) {
            ramp = (ramp * 2 < kReadBurst) ? ramp * 2 : kReadBurst;
        }
        return n;
    };
    const uint32_t depth = get_local_cb_interface(cb_id_src).fifo_num_pages;
    uint32_t filled_in_cycle = 0;
    for (uint32_t done = 0; done < num_pages;) {
        uint32_t n = next_burst();
        const uint32_t remaining = num_pages - done;
        const uint32_t until_wrap = depth - filled_in_cycle;
        if (n > remaining) {
            n = remaining;
        }
        if (n > until_wrap) {
            n = until_wrap;
        }
        dfb_src.reserve_back(n);
        for (uint32_t k = 0; k < n; ++k) {
            noc.async_read(src, dfb_src, page_bytes, {.page_id = next_page()}, {.offset_bytes = k * page_bytes});
        }
        noc.async_read_barrier();
        dfb_src.push_back(n);
        filled_in_cycle += n;
        if (filled_in_cycle == depth) {
            filled_in_cycle = 0;
        }
        done += n;
    }
#else
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_src.reserve_back(onepage);
        noc.async_read(src, dfb_src, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_src.push_back(onepage);
    }
#endif  // WORK_QUEUE
#endif
#endif
}
