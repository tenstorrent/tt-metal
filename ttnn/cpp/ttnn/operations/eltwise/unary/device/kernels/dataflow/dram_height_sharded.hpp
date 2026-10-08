// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRAM height-sharded unary: page order (SHARD_ROTATE) and work queue (WORK_QUEUE).
//
// Each shard sits in one bank, so pages are visited in rotated order: position q is slot q / num_shards of
// shard q % num_shards. Consecutive pages then come from different banks.
//
// The work queue cuts the rotated order into chunks. Worker w starts on chunk w and gets each next chunk from a
// scheduler in one core's writer: the worker writes kRequestTag | seq into its word of the scheduler's request
// table, and the scheduler answers (seq << kSeqShift) | chunk, or kDone, in the worker's reply semaphore. A reply
// counts only if its seq matches, so a stale value is never taken for a new chunk.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/eltwise/unary/device/kernels/dram_height_sharded_common.hpp"

namespace dram_hs {

constexpr uint32_t kRequestTag = 0xA5000000u;
constexpr uint32_t kSeqMask = 0xFFFu;
constexpr uint32_t kSeqShift = 20;
constexpr uint32_t kDone = 0xFFFFFu;  // also the chunk mask

struct RotatedPages {
    uint32_t shard_pages;
    uint32_t num_shards;
    uint32_t last_shard_pages;  // a short last shard has no page at slots >= last_shard_pages; they are skipped
    uint32_t slot = 0;
    uint32_t shard = 0;

    void seek(uint32_t q) {
        const uint32_t full = last_shard_pages * num_shards;
        if (q < full) {
            slot = q / num_shards;
            shard = q % num_shards;
        } else {
            slot = last_shard_pages + (q - full) / (num_shards - 1);
            shard = (q - full) % (num_shards - 1);
        }
    }

    uint32_t next() {
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
    }
};

// Cuts page groups at the CB wrap, so each group is contiguous in L1.
struct CbGroups {
    uint32_t depth;
    uint32_t pos = 0;

    uint32_t next(uint32_t want) {
        const uint32_t n = want < depth - pos ? want : depth - pos;
        pos = (pos + n == depth) ? 0 : pos + n;
        return n;
    }
};

inline uint32_t num_chunks(uint32_t total_pages, uint32_t chunk_pages) {
    return (total_pages + chunk_pages - 1) / chunk_pages;
}

// NoC address of l1_addr on the core whose coordinates are packed x | (y << 16).
inline uint64_t noc_addr(uint32_t xy, uint32_t l1_addr) { return get_noc_addr(xy & 0xFFFF, xy >> 16, l1_addr); }

inline volatile tt_l1_ptr uint32_t* semaphore(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}

// Reader: tells compute the chunk's tile count and the writer its (first position, count). Count 0 ends both.
inline void announce(uint32_t first, uint32_t count) {
    cb_reserve_back(kCbComputeCount, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kCbComputeCount))[0] = count;
    cb_push_back(kCbComputeCount, 1);
    cb_reserve_back(kCbWriterChunk, 1);
    auto* chunk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kCbWriterChunk));
    chunk[0] = first;
    chunk[1] = count;
    cb_push_back(kCbWriterChunk, 1);
}

// Worker side, in the reader.
struct Client {
    uint64_t request_addr;  // this worker's word in the scheduler's request table
    volatile tt_l1_ptr uint32_t* reply = semaphore(kReplySemaphore);
    volatile tt_l1_ptr uint32_t* go = semaphore(kGoSemaphore);
    uint32_t seq = 0;
    bool started = false;

    // Sends a request if the scheduler has started; returns whether it did.
    bool try_request() {
        if (!started) {
            invalidate_l1_cache();
            started = *go == 1;
        }
        if (started) {
            seq = (seq + 1) & kSeqMask;
            seq += seq == 0;  // 0 is the table's empty value
            noc_inline_dw_write(request_addr, kRequestTag | seq);
        }
        return started;
    }

    void request() {
        while (!try_request()) {
        }
    }

    uint32_t receive() {
        uint32_t v;
        do {
            invalidate_l1_cache();
            v = *reply;
        } while ((v >> kSeqShift) != seq);
        return v & kDone;
    }
};

// Scheduler side, in one core's writer. Worker coordinates are common runtime args from coord_arg on.
struct Scheduler {
    uint32_t num_workers;
    uint32_t total_chunks;
    uint32_t coord_arg;
    uint32_t next_chunk = num_workers;  // chunks below num_workers are the workers' first ones
    uint32_t workers_done = 0;
    volatile tt_l1_ptr uint32_t* table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kCbRequestTable));
    uint32_t last_seq[kMaxWorkers] = {};

    uint64_t worker_addr(uint32_t k, uint32_t l1_addr) const {
        return noc_addr(get_common_arg_val<uint32_t>(coord_arg + k), l1_addr);
    }

    // Clears the request table, then lets the workers send requests.
    void start() {
        for (uint32_t k = 0; k < num_workers; ++k) {
            table[k] = 0;
        }
        for (uint32_t k = 0; k < num_workers; ++k) {
            noc_inline_dw_write(worker_addr(k, get_semaphore(kGoSemaphore)), 1);
        }
    }

    // Answers every new request: the next chunk, or kDone once all are handed out.
    void serve() {
        invalidate_l1_cache();
        for (uint32_t k = 0; k < num_workers; ++k) {
            const uint32_t v = table[k];
            const uint32_t seq = v & kSeqMask;
            if ((v & 0xFF000000u) != kRequestTag || seq == last_seq[k]) {
                continue;
            }
            last_seq[k] = seq;
            uint32_t chunk = kDone;
            if (next_chunk < total_chunks) {
                chunk = next_chunk++;
            } else {
                workers_done++;
            }
            noc_inline_dw_write(worker_addr(k, get_semaphore(kReplySemaphore)), (seq << kSeqShift) | chunk);
        }
    }

    bool finished() const { return workers_done >= num_workers; }
};

}  // namespace dram_hs
