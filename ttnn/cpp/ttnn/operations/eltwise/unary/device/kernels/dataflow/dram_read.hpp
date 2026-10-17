// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader and writer helpers for the unary DRAM read flows: the rotated page order, DFB-sized page groups and the
// work queue.

#pragma once

#include <algorithm>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/eltwise/unary/device/kernels/dram_read_common.hpp"

namespace dram_read {

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
            .shard_stride = get_arg(args::shard_stride),
            .num_shards = get_arg(args::num_shards),
            .last_shard_pages = get_arg(args::last_shard_pages),
            .shard_width = get_arg(args::shard_width),
            .row_pages = get_arg(args::row_pages)};
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

// Splits runs of pages into groups that stop at the end of the DFB, so each group is contiguous in L1.
struct DfbGroups {
    uint32_t depth;  // DFB entries
    uint32_t pos = 0;

    uint32_t next(uint32_t want) {
        const uint32_t n = std::min(want, depth - pos);
        pos = (pos + n == depth) ? 0 : pos + n;
        return n;
    }
};

#if WORK_QUEUE
// Work queue. Every core is a worker; the scheduler core's writer also hands out chunks.
//   request  worker -> its word in the scheduler's request table: seq | x << 12 | y << 22, the worker's NoC
//            coordinates for the reply and a sequence number that changes with every request (1 to kMaxSeq)
//   reply    scheduler -> the worker's reply semaphore, incremented by chunk + 1; the worker clears it before asking
// Each worker first clears its word in the table, waits for that write to land, then increments the scheduler's
// cleared semaphore; the scheduler serves once every worker has, so 0 means no request.
constexpr uint32_t kMaxSeq = 0xFFF;
constexpr uint32_t kNoChunk = 0xFFFFFFFE;  // reply when every chunk is out; kNoChunk + 1 still fits

inline volatile tt_l1_ptr uint32_t* l1_words(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}

// Writes one word to l1_addr on the core at NoC coordinates xy = x | (y << 16).
inline void write_word(const Noc& noc, uint32_t xy, uint32_t l1_addr, uint32_t value) {
    noc.inline_dw_write(UnicastEndpoint{}, value, {.noc_x = xy & 0xFFFF, .noc_y = xy >> 16, .addr = l1_addr});
}

#ifdef READ_BURST  // the reader
// Work queue, reader side. A worker starts on chunk worker_id, and asks for its next chunk before reading the current
// one, so the reply arrives while it reads. Each chunk goes on to compute and the writer.
struct QueueReader {
    const Noc& noc;
    uint32_t worker_id;
    uint32_t total_pages;
    uint32_t chunk_pages;
    uint32_t scheduler_xy;
    // This worker's NoC coordinates, shifted into place; on these chips they are the same on both NoCs.
    uint32_t request_coords = ((my_x[noc_index] & 0x3FF) | ((my_y[noc_index] & 0x3FF) << 10)) << 12;
    uint32_t table_slot = DataflowBuffer(dfb::requests).get_write_ptr() + sizeof(uint32_t) * worker_id;
    Semaphore<> reply{sem::reply};
    uint32_t seq = 0;

    static QueueReader from_args(const Noc& noc) {
        return {
            .noc = noc,
            .worker_id = get_arg(args::start_id),
            .total_pages = get_arg(args::num_pages),
            .chunk_pages = get_arg(args::chunk_pages),
            .scheduler_xy = get_arg(args::scheduler_xy)};
    }

    // Calls read(first, count) for every chunk this worker gets.
    template <typename Read>
    void for_each_chunk(Read read) {
        write_word(noc, scheduler_xy, table_slot, 0);
        noc.async_write_barrier();
        Semaphore<>(sem::cleared).up(noc, scheduler_xy & 0xFFFF, scheduler_xy >> 16, 1);
        const uint32_t num_chunks = tt::data_movement::common::div_up(total_pages, chunk_pages);
        for (uint32_t chunk = worker_id; chunk != kNoChunk; chunk = wait_reply()) {
            request();
            if (chunk < num_chunks) {
                const uint32_t first = chunk * chunk_pages;
                const uint32_t count = std::min(chunk_pages, total_pages - first);
                hand_on(first, count);
                read(first, count);
            }
        }
        noc.async_write_barrier();  // the requests are NoC writes
        hand_on(0, 0);              // tells compute and the writer that this worker is done
    }

    void request() {
        reply.set(0);
        seq = seq % kMaxSeq + 1;
        write_word(noc, scheduler_xy, table_slot, request_coords | seq);
    }

    // Waits for the reply to the latest request and returns its chunk.
    uint32_t wait_reply() const {
        uint32_t value;
        do {
            value = reply.value();
        } while (value == 0);
        return value - 1;
    }

    // Passes a chunk on: its tile count to compute, its first position and page count to the writer.
    static void hand_on(uint32_t first, uint32_t count) {
        DataflowBuffer to_compute(dfb::chunk_count);
        to_compute.reserve_back(1);
        l1_words(to_compute.get_write_ptr())[0] = count;
        to_compute.push_back(1);
        DataflowBuffer to_writer(dfb::chunk_range);
        to_writer.reserve_back(1);
        auto* chunk = l1_words(to_writer.get_write_ptr());
        chunk[0] = first;
        chunk[1] = count;
        to_writer.push_back(1);
    }
};

#endif  // READ_BURST

#ifdef WRITE_BURST  // the writer
// Work queue, writer side. On the scheduler core the writer also answers requests while it waits for its own pages.
struct QueueWriter {
    const Noc& noc;
    bool is_scheduler;
    uint32_t num_chunks;
    uint32_t num_workers;
    uint32_t next_chunk = num_workers;  // chunks below num_workers are the workers' first chunks
    uint32_t workers_done = 0;
    bool serving = false;  // set once every worker has cleared its word in the table
    volatile tt_l1_ptr uint32_t* table = l1_words(DataflowBuffer(dfb::requests).get_read_ptr());
    Semaphore<> reply{sem::reply};
    uint32_t last_request[kMaxWorkers] = {};

    static QueueWriter from_args(const Noc& noc) {
        // The middle worker schedules, to keep request round trips short.
        return {
            .noc = noc,
            .is_scheduler = get_arg(args::start_id) == get_arg(args::num_workers) / 2,
            .num_chunks = tt::data_movement::common::div_up(get_arg(args::num_pages), get_arg(args::chunk_pages)),
            .num_workers = get_arg(args::num_workers)};
    }

    // Calls write(first, count) for every chunk the reader hands on, then keeps answering until every worker is done.
    template <typename Write>
    void for_each_chunk(Write write) {
        DataflowBuffer from_reader(dfb::chunk_range);
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

    // Answers every new request with the next chunk, or with kNoChunk once all chunks are out.
    void serve() {
        if (!serving) {
            serving = Semaphore<>(sem::cleared).value() >= num_workers;
            return;
        }
        invalidate_l1_cache();
        for (uint32_t k = 0; k < num_workers; ++k) {
            const uint32_t word = table[k];
            if (word == 0 || word == last_request[k]) {
                continue;
            }
            last_request[k] = word;
            uint32_t chunk = kNoChunk;
            if (next_chunk < num_chunks) {
                chunk = next_chunk++;
            } else {
                workers_done++;
            }
            reply.up(noc, (word >> 12) & 0x3FF, word >> 22, chunk + 1);
        }
    }
};
#endif  // WRITE_BURST
#endif  // WORK_QUEUE

}  // namespace dram_read
