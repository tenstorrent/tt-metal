// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader and writer helpers for DRAM-sharded unary: rotated page order and work queue.

#pragma once

#include <algorithm>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/eltwise/unary/device/kernels/dram_sharded_common.hpp"

namespace dram_shard {

// A request is kRequestTag | seq; a reply is (seq << kSeqShift) | chunk and counts only if seq matches.
constexpr uint32_t kRequestTag = 0xA5000000u;
constexpr uint32_t kTagMask = 0xFF000000u;
constexpr uint32_t kSeqMask = 0xFFFu;
constexpr uint32_t kSeqShift = 20;
constexpr uint32_t kDone = 0xFFFFFu;  // also the chunk mask

inline volatile tt_l1_ptr uint32_t* l1_words(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}

// Writes val to l1_addr on the core whose coordinates are packed x | (y << 16).
inline void write_word(const Noc& noc, uint32_t xy, uint32_t l1_addr, uint32_t val) {
    noc.inline_dw_write(UnicastEndpoint{}, val, {.noc_x = xy & 0xFFFF, .noc_y = xy >> 16, .addr = l1_addr});
}

// Visits slot 0 of every shard, then slot 1, and so on, so consecutive pages come from different banks.
struct RotatedPages {
    uint32_t shard_stride;  // page ids between shards
    uint32_t num_shards;
    uint32_t last_shard_pages;  // slots in the last, possibly short, shard
    uint32_t shard_width;       // pages per shard row
    uint32_t row_pages;         // pages per tensor row
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

    void seek(uint32_t q) {
        const uint32_t full = last_shard_pages * num_shards;
        if (q < full) {
            slot = q / num_shards;
            shard = q % num_shards;
        } else {
            slot = last_shard_pages + (q - full) / (num_shards - 1);
            shard = (q - full) % (num_shards - 1);
        }
        col = slot % shard_width;
        slot_offset = (slot / shard_width) * row_pages + col;
        shard_offset = shard * shard_stride;
    }

    // Slot k of shard s is page s * shard_stride + (k / shard_width) * row_pages + k % shard_width.
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

// Cuts page groups at the CB wrap, so each is contiguous in L1.
struct CbGroups {
    uint32_t depth;
    uint32_t pos = 0;

    uint32_t next(uint32_t want) {
        const uint32_t n = std::min(want, depth - pos);
        pos = (pos + n == depth) ? 0 : pos + n;
        return n;
    }
};

// Work queue, reader side: each core is a worker that starts on chunk worker_id.
struct QueueReader {
    const Noc& noc;
    uint32_t worker_id;
    uint32_t total_pages;
    uint32_t chunk_pages;
    uint32_t scheduler_xy;
    uint32_t seq = 0;
    bool started = false;

    static QueueReader from_args(const Noc& noc) {
        return {
            .noc = noc,
            .worker_id = get_arg_val<uint32_t>(kArgSplit),
            .total_pages = get_arg_val<uint32_t>(kArgSplit + 1),
            .chunk_pages = get_arg_val<uint32_t>(kArgQueue),
            .scheduler_xy = get_arg_val<uint32_t>(kArgQueue + 1)};
    }

    // Calls process(first, count) per chunk, asking for the next chunk before reading this one.
    template <typename Process>
    void for_each_chunk(Process process) {
        const uint32_t num_chunks = tt::data_movement::common::div_up(total_pages, chunk_pages);
        for (uint32_t chunk = worker_id; chunk != kDone; chunk = receive()) {
            const bool asked = try_request();  // before the scheduler starts, ask after reading
            if (chunk < num_chunks) {
                const uint32_t first = chunk * chunk_pages;
                const uint32_t count = std::min(chunk_pages, total_pages - first);
                announce(first, count);
                process(first, count);
            }
            while (!asked && !try_request()) {
            }
        }
        noc.async_write_barrier();  // flush request writes
        announce(0, 0);
    }

    // Requests a chunk if the scheduler has started; returns whether it did.
    bool try_request() {
        started = started || Semaphore<>(kGoSemaphore).value() == 1;
        if (started) {
            seq = (seq + 1) & kSeqMask;
            seq += seq == 0;  // 0 means empty
            const uint32_t table = DataflowBuffer(kCbRequestTable).get_write_ptr();
            write_word(noc, scheduler_xy, table + sizeof(uint32_t) * worker_id, kRequestTag | seq);
        }
        return started;
    }

    uint32_t receive() const {
        const Semaphore<> reply(kReplySemaphore);
        uint32_t v;
        do {
            v = reply.value();
        } while ((v >> kSeqShift) != seq);
        return v & kDone;
    }

    static void announce(uint32_t first, uint32_t count) {
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

// Work queue, writer side; on one core it also runs the scheduler.
struct QueueWriter {
    const Noc& noc;
    bool is_scheduler;
    uint32_t num_chunks;
    uint32_t num_workers;
    uint32_t coord_arg;                 // worker coordinates are common runtime args from here on
    uint32_t next_chunk = num_workers;  // chunks below num_workers are first chunks
    uint32_t workers_done = 0;
    volatile tt_l1_ptr uint32_t* table = l1_words(DataflowBuffer(kCbRequestTable).get_write_ptr());
    uint32_t last_seq[kMaxWorkers] = {};

    static QueueWriter from_args(const Noc& noc, uint32_t coord_arg) {
        const uint32_t total_pages = get_arg_val<uint32_t>(kArgSplit + 1);
        const uint32_t chunk_pages = get_arg_val<uint32_t>(kArgQueue);
        return {
            .noc = noc,
            .is_scheduler = get_arg_val<uint32_t>(kArgSplit) != 0,
            .num_chunks = tt::data_movement::common::div_up(total_pages, chunk_pages),
            .num_workers = get_arg_val<uint32_t>(kArgQueue + 1),
            .coord_arg = coord_arg};
    }

    // Calls process(first, count) per chunk from the reader, then serves requests until all workers are done.
    template <typename Process>
    void for_each_chunk(Process process) {
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
            process(first, count);
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

    // On the scheduler core, clears the request table and lets the workers request.
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

    // Replies to new requests with the next chunk, or kDone when none are left.
    void serve() {
        invalidate_l1_cache();
        for (uint32_t k = 0; k < num_workers; ++k) {
            const uint32_t v = table[k];
            const uint32_t seq = v & kSeqMask;
            if ((v & kTagMask) != kRequestTag || seq == last_seq[k]) {
                continue;
            }
            last_seq[k] = seq;
            uint32_t chunk = kDone;
            if (next_chunk < num_chunks) {
                chunk = next_chunk++;
            } else {
                workers_done++;
            }
            write_word(noc, worker_xy(k), get_semaphore(kReplySemaphore), (seq << kSeqShift) | chunk);
        }
    }

    uint32_t worker_xy(uint32_t k) const { return get_common_arg_val<uint32_t>(coord_arg + k); }
};

}  // namespace dram_shard
