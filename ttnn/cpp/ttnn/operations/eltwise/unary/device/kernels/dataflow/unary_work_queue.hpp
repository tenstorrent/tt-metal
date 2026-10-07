// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Work queue for DRAM height-sharded unary ops (WORK_QUEUE=1).
//
// The tensor's pages are visited in the rotated order (slot by slot across the shards) and cut into
// fixed chunks of chunk_pages positions. Worker w starts on chunk w; every further chunk comes from a
// scheduler that runs inside one core's writer. There are no NoC atomics:
//   worker:    posts kRequestTag | seq into its slot of the request table (CB 7, same L1 address on
//              every core) on the scheduler core, then polls its reply semaphore;
//   scheduler: answers (seq << kSeqShift) | chunk into the worker's reply semaphore; kDone stops it.
// A reply only counts when its seq matches the request, so a stale value cannot be taken for a chunk.
// Before any request the scheduler clears the table and raises every worker's go semaphore.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace unary_wq {

constexpr uint32_t kCbComputeCount = 4;  // reader -> compute: tile count of the next chunk, 0 = stop
constexpr uint32_t kCbWriterChunk = 5;   // reader -> writer: (first position, page count), count 0 = stop
constexpr uint32_t kCbRequestTable = 7;  // one word per worker, read on the scheduler core
constexpr uint32_t kReplySemaphore = 0;
constexpr uint32_t kGoSemaphore = 1;

constexpr uint32_t kRequestTag = 0xA5000000u;
constexpr uint32_t kSeqMask = 0xFFFu;
constexpr uint32_t kSeqShift = 20;
constexpr uint32_t kChunkMask = 0xFFFFFu;
constexpr uint32_t kDone = 0xFFFFFu;
constexpr uint32_t kMaxWorkers = 144;  // the factory keeps larger grids on the static split

// Rotated order: position q visits slot q / num_shards of shard q % num_shards, except that a short last
// shard has no page at slots >= last_shard_pages; those positions are skipped, as in SHARD_ROTATE.
struct RotatedPages {
    uint32_t shard_pages;
    uint32_t num_shards;
    uint32_t last_shard_pages;
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

// Chunk c covers positions [c * chunk_pages, min((c + 1) * chunk_pages, total_pages)).
inline uint32_t num_chunks(uint32_t total_pages, uint32_t chunk_pages) {
    return (total_pages + chunk_pages - 1) / chunk_pages;
}

// Worker side of the protocol (runs in the reader).
struct Client {
    uint64_t request_noc_addr;
    volatile tt_l1_ptr uint32_t* reply;
    volatile tt_l1_ptr uint32_t* go;
    uint32_t seq = 0;
    bool go_seen = false;

    bool go_raised() {
        if (!go_seen) {
            invalidate_l1_cache();
            go_seen = (*go == 1);
        }
        return go_seen;
    }

    void request() {
        while (!go_raised()) {
        }
        seq = (seq + 1) & kSeqMask;
        if (seq == 0) {
            seq = 1;
        }
        noc_inline_dw_write(request_noc_addr, kRequestTag | seq);
    }

    uint32_t receive() {
        uint32_t v;
        do {
            invalidate_l1_cache();
            v = *reply;
        } while ((v >> kSeqShift) != seq);
        return v & kChunkMask;
    }
};

// Scheduler side (runs in the writer of one core). Worker NoC coordinates are common runtime args,
// packed x | (y << 16), starting at coord_arg; reply addresses are built from them on use so that only
// one word per worker stays on the (small) BRISC stack.
struct Scheduler {
    volatile tt_l1_ptr uint32_t* table;
    uint32_t num_workers;
    uint32_t total_chunks;
    uint32_t next_chunk;
    uint32_t coord_arg;
    uint32_t workers_done = 0;
    uint32_t last_seq[kMaxWorkers];

    uint64_t worker_addr(uint32_t k, uint32_t l1_addr) const {
        const uint32_t xy = get_common_arg_val<uint32_t>(coord_arg + k);
        return get_noc_addr(xy & 0xFFFF, xy >> 16, l1_addr);
    }

    void start() {
        for (uint32_t k = 0; k < num_workers; ++k) {
            table[k] = 0;
            last_seq[k] = 0;
        }
        const uint32_t go_l1 = get_semaphore(kGoSemaphore);
        for (uint32_t k = 0; k < num_workers; ++k) {
            noc_inline_dw_write(worker_addr(k, go_l1), 1);
        }
    }

    // One pass over the table: answer every new request.
    void serve() {
        invalidate_l1_cache();
        const uint32_t reply_l1 = get_semaphore(kReplySemaphore);
        for (uint32_t k = 0; k < num_workers; ++k) {
            const uint32_t v = table[k];
            if ((v & 0xFF000000u) != kRequestTag) {
                continue;
            }
            const uint32_t seq = v & kSeqMask;
            if (seq == last_seq[k]) {
                continue;
            }
            last_seq[k] = seq;
            uint32_t chunk = kDone;
            if (next_chunk < total_chunks) {
                chunk = next_chunk++;
            } else {
                workers_done++;
            }
            noc_inline_dw_write(worker_addr(k, reply_l1), (seq << kSeqShift) | chunk);
        }
    }

    bool finished() const { return workers_done >= num_workers; }
};

}  // namespace unary_wq
