// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include "sparse_push_constants.hpp"

// The L1 loops of the plan's steps (see the overview in sparse_push_plan.cpp); every worker runs its 1/5 share.
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

inline void wait_at_least(uint32_t sem_addr, uint32_t v) {
    auto* p = reinterpret_cast<volatile uint32_t*>(sem_addr);
    do {
        fence();  // drop the cached line (Blackhole's small write-through L1 cache)
    } while (*p < v);
    fence();
}

inline void post(uint32_t sem_addr, uint32_t v) {
    fence();
    *reinterpret_cast<volatile uint32_t*>(sem_addr) = v;
}

// Barrier of the core's five workers: report `progress`, then wait until every worker has reported it.
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
    for (uint32_t i = begin; i < end; ++i) {
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

// Sets the bits of every row the five workers handed this worker, and flags their chunks dirty (one byte per chunk,
// so concurrent writers need no atomics). Returns the largest row + 1 (0 = none).
inline uint32_t scatter(
    const uint32_t* buckets,
    uint32_t bucket_cap,
    const uint32_t* bucketed,
    uint32_t* bitmap,
    uint8_t* dirty_chunks,
    uint32_t worker) {
    // Top-k rows come mostly ascending, so runs of indices share a word and a chunk: keep the current word's bits
    // in a register and store once per run (L1 stores are the scatter's bottleneck).
    uint32_t row_end = 0;
    uint32_t word = kSentinel, mask = 0, chunk = kSentinel;
    for (uint32_t from = 0; from < kWorkers; ++from) {
        const uint32_t* rows = buckets + (from * kWorkers + worker) * bucket_cap;
        const uint32_t count = bucketed[from * kWorkers + worker];
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t row = rows[i];
            const uint32_t row_word = row / kRowsPerWord;
            if (row_word != word) {
                if (word != kSentinel) {
                    bitmap[word] |= mask;
                }
                word = row_word;
                mask = 0;
                const uint32_t row_chunk = row / kChunkRows;
                if (row_chunk != chunk) {
                    dirty_chunks[row_chunk] = 1;
                    chunk = row_chunk;
                }
            }
            mask |= 1u << (row % kRowsPerWord);
            row_end = std::max(row_end, row + 1);
        }
    }
    if (word != kSentinel) {
        bitmap[word] |= mask;
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
// chunk, so a chunk every core touched is not left to one worker. Unrolled for 16-word chunks over five workers: it
// reads each copy's list entry once, where a per-word loop re-walks the list (measured ~20% slower plan).
template <uint32_t cores>
inline void or_copies(
    const uint32_t* scratch,
    const uint16_t* lists,
    const uint32_t* copies,
    uint32_t my_chunks,
    uint32_t core_idx,
    uint32_t* bitmap,
    uint32_t worker) {
    static_assert(
        kChunkWords == 16 && kWorkers == 5, "worker 0 takes words 0, 5, 10, 15; worker w words w, w + 5, w + 10");
    const bool has_fourth_word = worker == 0;
    for (uint32_t my_chunk = 0; my_chunk < my_chunks; ++my_chunk) {
        const uint32_t count = copies[my_chunk];
        const uint16_t* list = lists + my_chunk * cores;
        const uint32_t* first_word = scratch + my_chunk * cores * kChunkWords + worker;
        uint32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t* copy = first_word + list[i] * kChunkWords;
            acc0 |= copy[0];
            acc1 |= copy[kWorkers];
            acc2 |= copy[2 * kWorkers];
            if (has_fourth_word) {
                acc3 |= copy[3 * kWorkers];
            }
        }
        uint32_t* dst = bitmap + (core_idx + my_chunk * cores) * kChunkWords + worker;
        dst[0] = acc0;
        dst[kWorkers] = acc1;
        dst[2 * kWorkers] = acc2;
        if (has_fourth_word) {
            dst[3 * kWorkers] = acc3;
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
#pragma GCC unroll 2  // tuning knob
    for (uint32_t i = begin; i < end; ++i) {
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
