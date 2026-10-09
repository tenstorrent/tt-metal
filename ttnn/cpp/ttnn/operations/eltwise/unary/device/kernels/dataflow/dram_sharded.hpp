// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRAM-sharded unary, reader and writer side: page order (SHARD_ROTATE) and work queue (WORK_QUEUE).
//
// Position q is slot q / num_shards of shard q % num_shards, so consecutive pages hit different banks.
// Slot k of shard s is page s * shard_stride + (k / shard_width) * row_pages + k % shard_width.
//
// Work queue: worker w starts on chunk w and asks a scheduler in one core's writer for the next. It writes
// kRequestTag | seq to its request-table word; the scheduler replies (seq << kSeqShift) | chunk, or kDone, to its
// reply semaphore. A reply counts only if seq matches.

#pragma once

#include <algorithm>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/eltwise/unary/device/kernels/dram_sharded_common.hpp"

namespace dram_shard {

constexpr uint32_t kRequestTag = 0xA5000000u;
constexpr uint32_t kTagMask = 0xFF000000u;
constexpr uint32_t kSeqMask = 0xFFFu;
constexpr uint32_t kSeqShift = 20;
constexpr uint32_t kDone = 0xFFFFFu;  // also the chunk mask

inline volatile tt_l1_ptr uint32_t* l1_words(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}

// NoC address of l1_addr on the core whose coordinates are packed x | (y << 16).
inline uint64_t noc_addr(uint32_t xy, uint32_t l1_addr) { return get_noc_addr(xy & 0xFFFF, xy >> 16, l1_addr); }

// ---- Page order ----

struct RotatedPages {
    uint32_t shard_stride;  // page ids between shards
    uint32_t num_shards;
    uint32_t last_shard_pages;  // slots in the last shard; later slots are skipped there
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

// ---- Work queue, reader side: one worker per core ----

struct QueueReader {
    uint32_t worker_id;
    uint32_t total_pages;
    uint32_t chunk_pages;
    uint64_t request_addr;  // this worker's request-table word
    uint32_t seq = 0;
    bool started = false;

    static QueueReader from_args() {
        const uint32_t worker_id = get_arg_val<uint32_t>(kArgSplit);
        return {
            .worker_id = worker_id,
            .total_pages = get_arg_val<uint32_t>(kArgSplit + 1),
            .chunk_pages = get_arg_val<uint32_t>(kArgQueue),
            .request_addr =
                noc_addr(get_arg_val<uint32_t>(kArgQueue + 1), get_write_ptr(kCbRequestTable) + 4 * worker_id)};
    }

    // Calls process(first, count) for each chunk this worker gets, requesting the next chunk before reading the
    // current one. Each chunk is announced to compute and the writer; count 0 ends both.
    template <typename Process>
    void for_each_chunk(Process process) {
        const uint32_t num_chunks = (total_pages + chunk_pages - 1) / chunk_pages;
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
        noc_async_write_barrier();  // flush request writes
        announce(0, 0);
    }

    // Requests a chunk if the scheduler has started; returns whether it did.
    bool try_request() {
        if (!started) {
            invalidate_l1_cache();
            started = *l1_words(get_semaphore(kGoSemaphore)) == 1;
        }
        if (started) {
            seq = (seq + 1) & kSeqMask;
            seq += seq == 0;  // 0 means empty
            noc_inline_dw_write(request_addr, kRequestTag | seq);
        }
        return started;
    }

    uint32_t receive() const {
        uint32_t v;
        do {
            invalidate_l1_cache();
            v = *l1_words(get_semaphore(kReplySemaphore));
        } while ((v >> kSeqShift) != seq);
        return v & kDone;
    }

    static void announce(uint32_t first, uint32_t count) {
        cb_reserve_back(kCbComputeCount, 1);
        l1_words(get_write_ptr(kCbComputeCount))[0] = count;
        cb_push_back(kCbComputeCount, 1);
        cb_reserve_back(kCbWriterChunk, 1);
        auto* chunk = l1_words(get_write_ptr(kCbWriterChunk));
        chunk[0] = first;
        chunk[1] = count;
        cb_push_back(kCbWriterChunk, 1);
    }
};

// ---- Work queue, writer side: the scheduler runs in one core's writer ----

struct QueueWriter {
    bool is_scheduler;
    uint32_t num_chunks;
    uint32_t num_workers;
    uint32_t coord_arg;                 // worker coordinates are common runtime args from here on
    uint32_t next_chunk = num_workers;  // chunks below num_workers are first chunks
    uint32_t workers_done = 0;
    volatile tt_l1_ptr uint32_t* table = l1_words(get_write_ptr(kCbRequestTable));
    uint32_t last_seq[kMaxWorkers] = {};

    static QueueWriter from_args(uint32_t coord_arg) {
        const uint32_t total_pages = get_arg_val<uint32_t>(kArgSplit + 1);
        const uint32_t chunk_pages = get_arg_val<uint32_t>(kArgQueue);
        return {
            .is_scheduler = get_arg_val<uint32_t>(kArgSplit) != 0,
            .num_chunks = (total_pages + chunk_pages - 1) / chunk_pages,
            .num_workers = get_arg_val<uint32_t>(kArgQueue + 1),
            .coord_arg = coord_arg};
    }

    // Calls process(first, count) for each chunk the reader announces, until count 0. On the scheduler core, then
    // keeps answering requests until every worker is done.
    template <typename Process>
    void for_each_chunk(Process process) {
        start();
        while (true) {
            serve_until([] { return cb_pages_available_at_front(kCbWriterChunk, 1); });
            cb_wait_front(kCbWriterChunk, 1);
            const auto* chunk = l1_words(get_read_ptr(kCbWriterChunk));
            const uint32_t first = chunk[0];
            const uint32_t count = chunk[1];
            cb_pop_front(kCbWriterChunk, 1);
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
            noc_inline_dw_write(worker_addr(k, get_semaphore(kGoSemaphore)), 1);
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
            noc_inline_dw_write(worker_addr(k, get_semaphore(kReplySemaphore)), (seq << kSeqShift) | chunk);
        }
    }

    uint64_t worker_addr(uint32_t k, uint32_t l1_addr) const {
        return noc_addr(get_common_arg_val<uint32_t>(coord_arg + k), l1_addr);
    }
};

}  // namespace dram_shard
