// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader and writer helpers for the unary DRAM read flows: the rotated page order, CB-sized page groups and the
// work queue.

#pragma once

#include <algorithm>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/eltwise/unary/device/kernels/dram_read_common.hpp"

namespace dram_read {

// Work-queue messages are single words sent with NoC inline writes:
//   request  worker -> its slot in the scheduler's table   seq
//   reply    scheduler -> the worker's reply semaphore     (seq << kSeqShift) | chunk, chunk == kNoChunk when done
// seq runs from 1 to kMaxSeq and changes with every request, so a worker never mistakes the previous reply for a new
// one. The scheduler clears its table before it lets workers request, so 0 means no request.
constexpr uint32_t kMaxSeq = 0xFFF;
constexpr uint32_t kSeqShift = 20;
constexpr uint32_t kNoChunk = (1u << kSeqShift) - 1;  // also the mask of the chunk field

inline volatile tt_l1_ptr uint32_t* l1_words(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}

// Writes one word to l1_addr on the core at NoC coordinates xy = x | (y << 16).
inline void write_word(const Noc& noc, uint32_t xy, uint32_t l1_addr, uint32_t value) {
    noc.inline_dw_write(UnicastEndpoint{}, value, {.noc_x = xy & 0xFFFF, .noc_y = xy >> 16, .addr = l1_addr});
}

// The order in which a flow reads and writes pages: slot 0 of every shard, then slot 1, and so on, so consecutive
// pages come from different banks. Slot k of shard s is page s * shard_stride + (k / shard_width) * row_pages +
// k % shard_width. An interleaved tensor is one shard, so its order is the page order.
struct RotatedPages {
    uint32_t shard_stride;  // pages between the starts of two shards
    uint32_t num_shards;
    uint32_t last_shard_pages;  // slots in the last shard, which may be short
    uint32_t shard_width;       // pages per shard row
    uint32_t row_pages;         // pages per tensor row
    // Cursor: the next page is shard_offset + slot_offset.
    uint32_t slot = 0;
    uint32_t shard = 0;
    uint32_t col = 0;           // slot % shard_width
    uint32_t slot_offset = 0;   // page id of the slot in shard 0
    uint32_t shard_offset = 0;  // shard * shard_stride

    static RotatedPages from_args() {
        return {
            .shard_stride = get_arg_val<uint32_t>(kArgPageOrder),
            .num_shards = get_arg_val<uint32_t>(kArgPageOrder + 1),
            .last_shard_pages = get_arg_val<uint32_t>(kArgPageOrder + 2),
            .shard_width = get_arg_val<uint32_t>(kArgPageOrder + 3),
            .row_pages = get_arg_val<uint32_t>(kArgPageOrder + 4)};
    }

    // Moves the cursor to position pos of the order. Past the last shard's slots, the slots have one shard fewer.
    void seek(uint32_t pos) {
        const uint32_t full = last_shard_pages * num_shards;
        if (pos < full) {
            slot = pos / num_shards;
            shard = pos % num_shards;
        } else {
            slot = last_shard_pages + (pos - full) / (num_shards - 1);
            shard = (pos - full) % (num_shards - 1);
        }
        col = slot % shard_width;
        slot_offset = (slot / shard_width) * row_pages + col;
        shard_offset = shard * shard_stride;
    }

    // Returns the page at the cursor and moves the cursor on.
    uint32_t next() {
        if (shard == num_shards - 1 && slot >= last_shard_pages) {
            next_slot();
        }
        const uint32_t page = shard_offset + slot_offset;
        if (++shard == num_shards) {
            next_slot();
        } else {
            shard_offset += shard_stride;
        }
        return page;
    }

    void next_slot() {
        shard = 0;
        shard_offset = 0;
        slot++;
        if (++col == shard_width) {
            col = 0;
            slot_offset += row_pages - shard_width + 1;
        } else {
            slot_offset++;
        }
    }
};

// Splits runs of pages into groups that stop at the end of the CB, so each group is contiguous in L1.
struct CbGroups {
    uint32_t depth;  // CB pages
    uint32_t pos = 0;

    uint32_t next(uint32_t want) {
        const uint32_t n = std::min(want, depth - pos);
        pos = (pos + n == depth) ? 0 : pos + n;
        return n;
    }
};

// Work queue, reader side. Every core is a worker: it starts on chunk worker_id, and asks for its next chunk before
// reading the current one, so the reply arrives while it reads. Each chunk goes on to compute and the writer.
struct QueueReader {
    const Noc& noc;
    uint32_t worker_id;
    uint32_t total_pages;
    uint32_t chunk_pages;
    uint32_t scheduler_xy;
    uint32_t seq = 0;
    bool scheduler_started = false;

    static QueueReader from_args(const Noc& noc) {
        return {
            .noc = noc,
            .worker_id = get_arg_val<uint32_t>(kArgFirstPage),
            .total_pages = get_arg_val<uint32_t>(kArgNumPages),
            .chunk_pages = get_arg_val<uint32_t>(kArgChunkPages),
            .scheduler_xy = get_arg_val<uint32_t>(kArgSchedulerXy)};
    }

    // Calls read(first, count) for every chunk this worker gets.
    template <typename Read>
    void for_each_chunk(Read read) {
        const uint32_t num_chunks = tt::data_movement::common::div_up(total_pages, chunk_pages);
        for (uint32_t chunk = worker_id; chunk != kNoChunk; chunk = wait_reply()) {
            // Until the scheduler starts, the request has to wait until after the read.
            const bool asked = try_request();
            if (chunk < num_chunks) {
                const uint32_t first = chunk * chunk_pages;
                const uint32_t count = std::min(chunk_pages, total_pages - first);
                hand_on(first, count);
                read(first, count);
            }
            while (!asked && !try_request()) {
            }
        }
        noc.async_write_barrier();  // the requests are NoC writes
        hand_on(0, 0);              // tells compute and the writer that this worker is done
    }

    // Sends a request if the scheduler has started; returns whether it did.
    bool try_request() {
        scheduler_started = scheduler_started || Semaphore<>(kGoSemaphore).value() == 1;
        if (scheduler_started) {
            seq = seq % kMaxSeq + 1;
            const uint32_t table = DataflowBuffer(kCbRequestTable).get_write_ptr();
            write_word(noc, scheduler_xy, table + sizeof(uint32_t) * worker_id, seq);
        }
        return scheduler_started;
    }

    // Waits for the reply to the latest request and returns its chunk.
    uint32_t wait_reply() const {
        const Semaphore<> reply(kReplySemaphore);
        uint32_t value;
        do {
            value = reply.value();
        } while ((value >> kSeqShift) != seq);
        return value & kNoChunk;
    }

    // Passes a chunk on: its tile count to compute, its first position and page count to the writer.
    static void hand_on(uint32_t first, uint32_t count) {
        DataflowBuffer to_compute(kCbComputeCount);
        to_compute.reserve_back(1);
        l1_words(to_compute.get_write_ptr())[0] = count;
        to_compute.push_back(1);
        DataflowBuffer to_writer(kCbWriterChunk);
        to_writer.reserve_back(1);
        auto* chunk = l1_words(to_writer.get_write_ptr());
        chunk[0] = first;
        chunk[1] = count;
        to_writer.push_back(1);
    }
};

// Work queue, writer side. On the scheduler core the writer also answers requests while it waits for its own pages.
struct QueueWriter {
    const Noc& noc;
    bool is_scheduler;
    uint32_t num_chunks;
    uint32_t num_workers;
    uint32_t worker_xy_arg;             // worker coordinates are common runtime args from this index on
    uint32_t next_chunk = num_workers;  // chunks below num_workers are the workers' first chunks
    uint32_t workers_done = 0;
    volatile tt_l1_ptr uint32_t* table = l1_words(DataflowBuffer(kCbRequestTable).get_write_ptr());
    uint32_t last_seq[kMaxWorkers] = {};

    static QueueWriter from_args(const Noc& noc, uint32_t worker_xy_arg) {
        return {
            .noc = noc,
            .is_scheduler = get_arg_val<uint32_t>(kArgIsScheduler) != 0,
            .num_chunks = tt::data_movement::common::div_up(
                get_arg_val<uint32_t>(kArgNumPages), get_arg_val<uint32_t>(kArgChunkPages)),
            .num_workers = get_arg_val<uint32_t>(kArgNumWorkers),
            .worker_xy_arg = worker_xy_arg};
    }

    // Calls write(first, count) for every chunk the reader hands on, then keeps answering until every worker is done.
    template <typename Write>
    void for_each_chunk(Write write) {
        start();
        DataflowBuffer from_reader(kCbWriterChunk);
        while (true) {
            serve_until([&] { return from_reader.pages_available_at_front(1); });
            from_reader.wait_front(1);
            const auto* chunk = l1_words(from_reader.get_read_ptr());
            const uint32_t first = chunk[0];
            const uint32_t count = chunk[1];
            from_reader.pop_front(1);
            if (count == 0) {
                break;
            }
            write(first, count);
        }
        serve_until([&] { return workers_done >= num_workers; });
    }

    // On the scheduler core, answers requests until ready() holds; elsewhere returns at once.
    template <typename Ready>
    void serve_until(Ready ready) {
        while (is_scheduler && !ready()) {
            serve();
        }
    }

    // On the scheduler core, clears the request table, then lets the workers request.
    void start() {
        if (!is_scheduler) {
            return;
        }
        for (uint32_t k = 0; k < num_workers; ++k) {
            table[k] = 0;
        }
        for (uint32_t k = 0; k < num_workers; ++k) {
            write_word(noc, worker_xy(k), get_semaphore(kGoSemaphore), 1);
        }
    }

    // Answers every new request with the next chunk, or with kNoChunk once all chunks are out.
    void serve() {
        invalidate_l1_cache();
        for (uint32_t k = 0; k < num_workers; ++k) {
            const uint32_t seq = table[k];
            if (seq == 0 || seq == last_seq[k]) {
                continue;
            }
            last_seq[k] = seq;
            uint32_t chunk = kNoChunk;
            if (next_chunk < num_chunks) {
                chunk = next_chunk++;
            } else {
                workers_done++;
            }
            write_word(noc, worker_xy(k), get_semaphore(kReplySemaphore), (seq << kSeqShift) | chunk);
        }
    }

    uint32_t worker_xy(uint32_t k) const { return get_common_arg_val<uint32_t>(worker_xy_arg + k); }
};

}  // namespace dram_read
