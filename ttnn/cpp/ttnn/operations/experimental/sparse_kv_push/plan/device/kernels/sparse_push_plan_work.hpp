// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif
#include "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/common/kernels/sparse_push_constants.hpp"

// The L1 loops of the plan's steps (see the overview in sparse_push_plan.cpp); the kWorkers RISC-Vs of a core each run
// a share. Built for the data movement kernel and for the TRISC kernel (sparse_push_plan_trisc.cpp).
namespace sparse_push_plan {

constexpr uint32_t kSentinel = 0xFFFFFFFFu;
constexpr uint32_t kRowsPerWord = sparse_push::plan_table::kRowsPerWord;
// Bitmap words are dealt to the workers by residue w % kWordResidues, a contiguous run of residues each.
constexpr uint32_t kWordResidues = 32;

// BRISC posts these on the step semaphore; the other workers wait for them.
enum Step : uint32_t { kStepScatter = 1, kStepReduce, kStepRanks, kStepBases };
// Each worker posts these on its done semaphore, in this order (its value only grows).
enum Progress : uint32_t {
    kBucketed = 1,
    kScattered,
    kMapped,
    kPushed,  // NCRISC only
    kListed,
    kOred,
    kRanked,
    kTotalled,
    kBased,
    kRewritten,
};

inline void fence() { asm volatile("fence" ::: "memory"); }

// The core's workers hand each other L1 data through local semaphores: plain L1 words, so the TRISCs can use them too.
inline void wait_at_least(uint32_t sem_addr, uint32_t v) {
    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    WAYPOINT("SPPW");  // a worker stuck here shows in the watcher log
    do {
        invalidate_l1_cache();
    } while (*sem < v);
    WAYPOINT("SPPD");
    fence();
}

inline void post(uint32_t sem_addr, uint32_t v) {
    fence();  // the worker's L1 writes land before the semaphore says they did
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr) = v;
    fence();
}

// Barrier of the core's workers: report `progress`, then wait until every worker has reported it.
template <typename DoneSem>
inline void sync_workers(DoneSem&& done_sem, uint32_t worker, uint32_t progress) {
    post(done_sem(worker), progress);
    for (uint32_t w = 0; w < kWorkers; ++w) {
        wait_at_least(done_sem(w), progress);
    }
}

// Worker `worker`'s share of n items is [share_begin(n, worker), share_begin(n, worker + 1)).
inline uint32_t share_begin(uint32_t n, uint32_t worker) { return n * worker / kWorkers; }

// The worker that sets the bits of bitmap word `word`: workers own contiguous runs of the residues
// word % kWordResidues, so the words of each worker are disjoint and the bitmap needs no atomics.
inline uint32_t word_owner(uint32_t word) { return (word % kWordResidues) * kWorkers / kWordResidues; }

// Sorts this worker's indices [begin, end) into one bucket per word owner, skipping the sentinel and any row past the
// cache. Bucket (worker, to) holds the rows `worker` hands to word owner `to`, at buckets + (worker * kWorkers + to) *
// bucket_cap; bucketed[worker * kWorkers + to] gets its length.
inline void bucket(
    const uint32_t* idx,
    uint32_t max_rows,
    uint32_t begin,
    uint32_t end,
    uint32_t* buckets,
    uint32_t bucket_cap,
    uint32_t* bucketed,
    uint32_t worker) {
    uint32_t* const first = buckets + worker * kWorkers * bucket_cap;
    uint32_t* bucket_end[kWorkers];
    for (uint32_t to = 0; to < kWorkers; ++to) {
        bucket_end[to] = first + to * bucket_cap;
    }
    // Rows loaded four at a time, so their L1 loads issue back to back.
    uint32_t i = begin;
    for (; i + 4 <= end; i += 4) {
        uint32_t row[4];
#pragma GCC unroll 4
        for (uint32_t u = 0; u < 4; ++u) {
            row[u] = idx[i + u];
        }
#pragma GCC unroll 4
        for (uint32_t u = 0; u < 4; ++u) {
            if (row[u] < max_rows) {
                *bucket_end[word_owner(row[u] / kRowsPerWord)]++ = row[u];
            }
        }
    }
    for (; i < end; ++i) {
        const uint32_t row = idx[i];
        if (row < max_rows) {
            *bucket_end[word_owner(row / kRowsPerWord)]++ = row;
        }
    }
    uint32_t* const lengths = bucketed + worker * kWorkers;
    for (uint32_t to = 0; to < kWorkers; ++to) {
        lengths[to] = bucket_end[to] - (first + to * bucket_cap);
    }
}

// Sets the bits of every row the workers handed this worker, and flags their chunks dirty (one byte per chunk,
// so concurrent writers need no atomics). Returns the largest row + 1 (0 = none).
inline uint32_t scatter(
    const uint32_t* buckets,
    uint32_t bucket_cap,
    const uint32_t* bucketed,
    uint32_t* bitmap,
    uint8_t* dirty_chunks,
    uint32_t worker) {
    // kScatterLanes rows per iteration with their bitmap loads issued back to back. Lanes hitting the same
    // word fold the earlier lanes' bits in, and the stores go in lane order, so the last store of a word holds them
    // all.
    constexpr uint32_t kScatterLanes = 4;
    uint32_t row_end = 0;
    for (uint32_t from = 0; from < kWorkers; ++from) {
        const uint32_t* rows = buckets + (from * kWorkers + worker) * bucket_cap;
        const uint32_t count = bucketed[from * kWorkers + worker];
        uint32_t i = 0;
        for (; i + kScatterLanes <= count; i += kScatterLanes) {
            uint32_t w[kScatterLanes], m[kScatterLanes], v[kScatterLanes];
#pragma GCC unroll 4
            for (uint32_t u = 0; u < kScatterLanes; ++u) {
                const uint32_t row = rows[i + u];
                w[u] = row / kRowsPerWord;
                m[u] = 1u << (row % kRowsPerWord);
                row_end = std::max(row_end, row + 1);
            }
#pragma GCC unroll 4
            for (uint32_t u = 0; u < kScatterLanes; ++u) {
                v[u] = bitmap[w[u]];
            }
#pragma GCC unroll 4
            for (uint32_t u = 0; u < kScatterLanes; ++u) {
                uint32_t bits = v[u] | m[u];
#pragma GCC unroll 4
                for (uint32_t j = 0; j < u; ++j) {
                    bits |= w[j] == w[u] ? m[j] : 0u;
                }
                bitmap[w[u]] = bits;
                dirty_chunks[w[u] / kChunkWords] = 1;
            }
        }
        for (; i < count; ++i) {
            const uint32_t row = rows[i];
            bitmap[row / kRowsPerWord] |= 1u << (row % kRowsPerWord);
            dirty_chunks[row / kChunkRows] = 1;
            row_end = std::max(row_end, row + 1);
        }
    }
    return row_end;
}

// Step 1 for one worker: bucket its share of the indices, wait for the bitmap to be zero and for every worker's
// buckets, then scatter the rows it owns. Returns the largest row + 1 it set.
template <uint32_t max_rows, typename DoneSem, typename SyncZeroedBitmap>
inline uint32_t scatter_step(
    const uint32_t* idx,
    uint32_t index_count,
    uint32_t* buckets,
    uint32_t bucket_cap,
    uint32_t* bucketed,
    uint32_t* bitmap,
    uint8_t* dirty_chunks,
    uint32_t worker,
    DoneSem&& done_sem,
    SyncZeroedBitmap&& sync_zeroed_bitmap) {
    bucket(
        idx,
        max_rows,
        share_begin(index_count, worker),
        share_begin(index_count, worker + 1),
        buckets,
        bucket_cap,
        bucketed,
        worker);
    sync_zeroed_bitmap();  // the bucketing overlaps BRISC's zeroing of the bitmap
    sync_workers(done_sem, worker, kBucketed);
    return scatter(buckets, bucket_cap, bucketed, bitmap, dirty_chunks, worker);
}

// For this worker's share of the reducers: map[reducer] gets bit k set when this core dirtied the reducer's k-th
// chunk (global chunk reducer + k * cores).
template <uint32_t cores, uint32_t total_chunks>
inline void map_dirty(const uint8_t* dirty_chunks, uint32_t* map, uint32_t worker) {
    for (uint32_t reducer = share_begin(cores, worker); reducer < share_begin(cores, worker + 1); ++reducer) {
        uint32_t map_word = 0;
        for (uint32_t reducer_chunk = 0, chunk = reducer; chunk < total_chunks; ++reducer_chunk, chunk += cores) {
            map_word |= static_cast<uint32_t>(dirty_chunks[chunk]) << reducer_chunk;
        }
        map[reducer] = map_word;
    }
}

// As a reducer: for each of this core's chunks, the list of cores that pushed a copy of it (bit my_chunk of
// column[core]) and their count. Scratch slot my_chunk * cores + core then holds that core's copy.
template <uint32_t cores>
inline void list_copies(
    const uint32_t* column, uint32_t my_chunks, uint16_t* lists, uint32_t* copies, uint32_t worker) {
    for (uint32_t my_chunk = worker; my_chunk < my_chunks; my_chunk += kWorkers) {
        uint16_t* list = lists + my_chunk * cores;
        uint32_t count = 0;
        for (uint32_t core = 0; core < cores; ++core) {
            if ((column[core] >> my_chunk) & 1u) {
                list[count++] = core;
            }
        }
        copies[my_chunk] = count;
    }
}

// As a reducer: ORs the received copies of each of this core's chunks into its bitmap. Split by word rather than by
// chunk, so a chunk every core touched is not left to one worker: worker w takes words w, w + kWorkers, ... of each
// chunk, and reads each copy's list entry once.
template <uint32_t cores>
inline void or_copies(
    const uint32_t* scratch,
    const uint16_t* lists,
    const uint32_t* copies,
    uint32_t my_chunks,
    uint32_t core_idx,
    uint32_t* bitmap,
    uint32_t worker) {
    constexpr uint32_t kMaxWords = (kChunkWords + kWorkers - 1) / kWorkers;
    const uint32_t words = (kChunkWords - worker + kWorkers - 1) / kWorkers;  // this worker's words of a chunk
    for (uint32_t my_chunk = 0; my_chunk < my_chunks; ++my_chunk) {
        const uint32_t count = copies[my_chunk];
        const uint16_t* list = lists + my_chunk * cores;
        const uint32_t* first_word = scratch + my_chunk * cores * kChunkWords + worker;
        uint32_t acc[kMaxWords] = {};
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t* copy = first_word + list[i] * kChunkWords;
#pragma GCC unroll 4
            for (uint32_t k = 0; k < kMaxWords; ++k) {
                if (k < words) {
                    acc[k] |= copy[k * kWorkers];
                }
            }
        }
        uint32_t* dst = bitmap + (core_idx + my_chunk * cores) * kChunkWords + worker;
        for (uint32_t k = 0; k < words; ++k) {
            dst[k * kWorkers] = acc[k];
        }
    }
}

// Calls visit(range, first_block, end_block) for each of this core's rank ranges: range = core_idx, core_idx + cores,
// ..., each range_blocks blocks, clipped to active_blocks.
template <uint32_t cores, uint32_t range_blocks, typename Visit>
inline void visit_ranges(uint32_t core_idx, uint32_t ranges, uint32_t active_blocks, Visit&& visit) {
    for (uint32_t range = core_idx; range < ranges; range += cores) {
        const uint32_t first_block = range * range_blocks;
        visit(range, first_block, std::min(first_block + range_blocks, active_blocks));
    }
}

// word_ranks[word]: the block owner's rows in the range before the word; partial[owner]: the owner's rows in the
// range. first_block is a multiple of num_devices, so block b's owner is b % num_devices.
template <uint32_t num_devices, uint32_t words_per_block>
inline void range_ranks(
    const uint32_t* bitmap,
    uint16_t* word_ranks,
    uint32_t* partial,
    uint32_t first_block,
    uint32_t end_block,
    uint32_t worker) {
    for (uint32_t owner = worker; owner < num_devices; owner += kWorkers) {
        uint32_t rows = 0;
        for (uint32_t block = first_block + owner; block < end_block; block += num_devices) {
#pragma GCC unroll 8  // tuning: words_per_block is 5 on GLM, so the per-block loop unrolls fully
            for (uint32_t word_in_block = 0; word_in_block < words_per_block; ++word_in_block) {
                const uint32_t word = block * words_per_block + word_in_block;
                word_ranks[word] = static_cast<uint16_t>(rows);
                rows += __builtin_popcount(bitmap[word]);
            }
        }
        partial[owner] = rows;
    }
}

// Partials become exclusive prefixes over the ranges, in place: partials[range][owner] = the owner's rows in all
// earlier ranges. totals[owner] gets the owner's rows in all ranges.
template <uint32_t num_devices>
inline void owner_prefixes(uint32_t* partials, uint32_t ranges, uint32_t* totals, uint32_t worker) {
    for (uint32_t owner = worker; owner < num_devices; owner += kWorkers) {
        uint32_t rows_before = 0;
        for (uint32_t range = 0; range < ranges; ++range) {
            const uint32_t range_rows = partials[range * owner_stride(num_devices) + owner];
            partials[range * owner_stride(num_devices) + owner] = rows_before;
            rows_before += range_rows;
        }
        totals[owner] = rows_before;
    }
}

// run_bases[owner]: where the owner's run starts in the compact buffer, after the padded runs of the lower owners.
template <uint32_t num_devices, uint32_t pages_per_packet, uint32_t banks>
inline void owner_bases(const uint32_t* totals, uint32_t* run_bases, uint32_t worker) {
    for (uint32_t owner = worker; owner < num_devices; owner += kWorkers) {
        uint32_t run_base = 0;
        for (uint32_t lower_owner = 0; lower_owner < owner; ++lower_owner) {
            run_base += sparse_push::padded_run_pages<pages_per_packet, banks>(totals[lower_owner]);
        }
        run_bases[owner] = run_base;
    }
}

// Rewrites idx [begin, end) in place from cache rows to compact-buffer pages: a row's rank within its owner is its
// range's offset + its word's rank + the needed rows below it in the word. A row past the cache (malformed top-k) was
// never planned and becomes the sentinel, so the plan stays inside its buffers.
template <
    uint32_t num_devices,
    uint32_t max_rows,
    uint32_t stripe_chunk,
    uint32_t range_blocks,
    uint32_t pages_per_packet,
    uint32_t banks>
inline void rewrite(
    uint32_t* __restrict__ idx,
    uint32_t begin,
    uint32_t end,
    const uint32_t* __restrict__ bitmap,
    const uint16_t* __restrict__ word_ranks,
    const uint32_t* __restrict__ range_offsets,
    const uint32_t* __restrict__ run_bases) {
    // kRewriteLanes indices per iteration, branch-free, so their independent L1 loads issue back to back
    // instead of each stalling the in-order core in turn.
    constexpr uint32_t kRewriteLanes = 4;
    auto page_of = [&](uint32_t row, uint32_t valid_row, uint32_t ro, uint32_t wr, uint32_t bits, uint32_t rb) {
        const uint32_t rank = ro + wr + __builtin_popcount(bits & ((1u << (valid_row % kRowsPerWord)) - 1u));
        const uint32_t page = sparse_push::compact_page<pages_per_packet, banks>(rb, rank);
        return page | (static_cast<uint32_t>(row < max_rows) - 1u);  // the sentinel past the cache
    };
    uint32_t i = begin;
    for (; i + kRewriteLanes <= end; i += kRewriteLanes) {
        uint32_t row[kRewriteLanes], r[kRewriteLanes], ro[kRewriteLanes], wr[kRewriteLanes], bits[kRewriteLanes],
            rb[kRewriteLanes];
#pragma GCC unroll 4
        for (uint32_t u = 0; u < kRewriteLanes; ++u) {
            row[u] = idx[i + u];
        }
#pragma GCC unroll 4
        for (uint32_t u = 0; u < kRewriteLanes; ++u) {
            r[u] = row[u] < max_rows ? row[u] : 0u;
            const uint32_t block = r[u] / stripe_chunk;
            const uint32_t word = r[u] / kRowsPerWord;
            const uint32_t owner = block % num_devices;
            ro[u] = range_offsets[(block / range_blocks) * owner_stride(num_devices) + owner];
            wr[u] = word_ranks[word];
            bits[u] = bitmap[word];
            rb[u] = run_bases[owner];
        }
#pragma GCC unroll 4
        for (uint32_t u = 0; u < kRewriteLanes; ++u) {
            idx[i + u] = page_of(row[u], r[u], ro[u], wr[u], bits[u], rb[u]);
        }
    }
    for (; i < end; ++i) {
        const uint32_t row = idx[i];
        if (row >= max_rows) {
            idx[i] = kSentinel;
            continue;
        }
        const uint32_t block = row / stripe_chunk;
        const uint32_t word = row / kRowsPerWord;
        const uint32_t owner = block % num_devices;
        const uint32_t rank = range_offsets[(block / range_blocks) * owner_stride(num_devices) + owner] +
                              word_ranks[word] + __builtin_popcount(bitmap[word] & ((1u << (row % kRowsPerWord)) - 1u));
        idx[i] = sparse_push::compact_page<pages_per_packet, banks>(run_bases[owner], rank);
    }
}

// Steps 4 and 5 for one worker: owner prefixes and totals, then run bases, then the rewrite of its share of idx.
template <
    uint32_t num_devices,
    uint32_t max_rows,
    uint32_t stripe_chunk,
    uint32_t range_blocks,
    uint32_t pages_per_packet,
    uint32_t banks,
    typename DoneSem>
inline void bases_and_rewrite(
    uint32_t* idx,
    uint32_t index_count,
    const uint32_t* bitmap,
    const uint16_t* word_ranks,
    uint32_t* partials,
    uint32_t* misc,
    uint32_t worker,
    DoneSem&& done_sem) {
    owner_prefixes<num_devices>(partials, misc[kMiscRanges], misc + kMiscTotals, worker);
    sync_workers(done_sem, worker, kTotalled);
    owner_bases<num_devices, pages_per_packet, banks>(misc + kMiscTotals, misc + kMiscBase, worker);
    sync_workers(done_sem, worker, kBased);
    rewrite<num_devices, max_rows, stripe_chunk, range_blocks, pages_per_packet, banks>(
        idx,
        share_begin(index_count, worker),
        share_begin(index_count, worker + 1),
        bitmap,
        word_ranks,
        partials,
        misc + kMiscBase);
    post(done_sem(worker), kRewritten);  // report only: the write-back waits for it
}

}  // namespace sparse_push_plan
