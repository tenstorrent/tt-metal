// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// L1 loops shared by the plan's five workers per core: BRISC (0), NCRISC (1), TRISCs (2..4). A worker waits for
// step k, does its share and reports k.

#pragma once

#include <cstdint>

#include "sparse_push_constants.hpp"

namespace sparse_push_plan {

constexpr uint32_t kSentinel = 0xFFFFFFFFu;

// Steps BRISC posts (kStep*) and progress values workers report, in order.
constexpr uint32_t kStepScatter = 1, kBucketed = 1, kScattered = 2;
constexpr uint32_t kMapped = 3, kPushed = 4;
constexpr uint32_t kStepOr = 5, kListed = 5, kOred = 6, kStepRange = 7;
constexpr uint32_t kStepBases = 8, kTotalled = 8, kBased = 9, kRewritten = 10;

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

inline uint32_t share_begin(uint32_t n, uint32_t w) { return n * w / kWorkers; }

// The worker that sets the bits of bitmap word w: workers own contiguous runs of the residues w % 32, so the
// words of each worker are disjoint and the bitmap needs no atomics.
inline uint32_t word_owner(uint32_t w) { return ((w & 31) * kWorkers) >> 5; }

// List (worker, to) holds the indices worker hands to word owner `to`, at buckets + (worker * kWorkers + to) * cap.
inline void bucket(
    const uint32_t* idx,
    uint32_t begin,
    uint32_t end,
    uint32_t* buckets,
    uint32_t cap,
    uint32_t* bucketed,
    uint32_t worker) {
    uint32_t* const first = buckets + worker * kWorkers * cap;
    uint32_t *p0 = first, *p1 = first + cap, *p2 = first + 2 * cap, *p3 = first + 3 * cap, *p4 = first + 4 * cap;
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t v = idx[i];
        if (v == kSentinel) {
            continue;
        }
        switch (word_owner(v >> 5)) {
            case 0: *p0++ = v; break;
            case 1: *p1++ = v; break;
            case 2: *p2++ = v; break;
            case 3: *p3++ = v; break;
            default: *p4++ = v; break;
        }
    }
    uint32_t* const out = bucketed + worker * kWorkers;
    out[0] = p0 - first;
    out[1] = p1 - (first + cap);
    out[2] = p2 - (first + 2 * cap);
    out[3] = p3 - (first + 3 * cap);
    out[4] = p4 - (first + 4 * cap);
}

// Dirty flags are one byte per chunk so concurrent writers need no atomics. Returns the largest row + 1 (0 = none).
inline uint32_t scatter(
    const uint32_t* buckets, uint32_t cap, const uint32_t* bucketed, uint32_t* bits, uint8_t* dirty, uint32_t worker) {
    // Top-k rows come mostly ascending, so runs of indices share a word and a chunk: keep the current word's bits
    // in a register and store once per run (L1 stores are the scatter's bottleneck).
    uint32_t row_end = 0;
    uint32_t word = kSentinel, mask = 0, chunk = kSentinel;
    for (uint32_t from = 0; from < kWorkers; ++from) {
        const uint32_t* list = buckets + (from * kWorkers + worker) * cap;
        const uint32_t n = bucketed[from * kWorkers + worker];
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t v = list[i];
            const uint32_t w = v >> 5;
            if (w != word) {
                if (word != kSentinel) {
                    bits[word] |= mask;
                }
                word = w;
                mask = 0;
                const uint32_t c = v >> kChunkShift;
                if (c != chunk) {
                    dirty[c] = 1;
                    chunk = c;
                }
            }
            mask |= 1u << (v & 31);
            row_end = v + 1 > row_end ? v + 1 : row_end;
        }
    }
    if (word != kSentinel) {
        bits[word] |= mask;
    }
    return row_end;
}

template <typename DoneSem, typename Zeroed>
inline uint32_t scatter_step(
    const uint32_t* idx,
    uint32_t n_idx,
    uint32_t* buckets,
    uint32_t cap,
    uint32_t* bucketed,
    uint32_t* bits,
    uint8_t* dirty,
    uint32_t worker,
    DoneSem&& done_sem,
    Zeroed&& zeroed) {
    bucket(idx, n_idx * worker / kWorkers, n_idx * (worker + 1) / kWorkers, buckets, cap, bucketed, worker);
    zeroed();  // the bucketing overlaps BRISC's zeroing of the bitmap
    post(done_sem(worker), kBucketed);
    for (uint32_t w = 0; w < kWorkers; ++w) {
        wait_at_least(done_sem(w), kBucketed);
    }
    return scatter(buckets, cap, bucketed, bits, dirty, worker);
}

template <uint32_t cores, uint32_t total_chunks>
inline void map_dirty(const uint8_t* dirty, uint32_t* map, uint32_t worker) {
    for (uint32_t r = share_begin(cores, worker); r < share_begin(cores, worker + 1); ++r) {
        uint32_t m = 0;
        for (uint32_t k = 0, ch = r; ch < total_chunks; ++k, ch += cores) {
            m |= static_cast<uint32_t>(dirty[ch]) << k;
        }
        map[r] = m;
    }
}

// Scratch slot k * cores + c holds core c's copy of this core's chunk k when bit k of column[c] is set.
template <uint32_t cores>
inline void list_copies(const uint32_t* column, uint32_t chunks, uint16_t* lists, uint32_t* copies, uint32_t worker) {
    for (uint32_t k = worker; k < chunks; k += kWorkers) {
        uint16_t* list = lists + k * cores;
        uint32_t n = 0;
        for (uint32_t c = 0; c < cores; ++c) {
            if ((column[c] >> k) & 1u) {
                list[n++] = c;
            }
        }
        copies[k] = n;
    }
}

// Split by word rather than by chunk, so a chunk every core touched is not left to one worker.
template <uint32_t cores>
inline void or_copies(
    const uint32_t* scratch,
    const uint16_t* lists,
    const uint32_t* copies,
    uint32_t chunks,
    uint32_t core_idx,
    uint32_t* bits,
    uint32_t worker) {
    static_assert(
        kChunkWords == 16 && kWorkers == 5, "worker 0 takes words 0, 5, 10, 15; worker w words w, w + 5, w + 10");
    const bool fourth = worker == 0;
    for (uint32_t k = 0; k < chunks; ++k) {
        const uint32_t n = copies[k];
        const uint16_t* list = lists + k * cores;
        const uint32_t* src = scratch + k * cores * kChunkWords + worker;
        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t* copy = src + list[i] * kChunkWords;
            a0 |= copy[0];
            a1 |= copy[kWorkers];
            a2 |= copy[2 * kWorkers];
            if (fourth) {
                a3 |= copy[3 * kWorkers];
            }
        }
        uint32_t* dst = bits + (core_idx + k * cores) * kChunkWords + worker;
        dst[0] = a0;
        dst[kWorkers] = a1;
        dst[2 * kWorkers] = a2;
        if (fourth) {
            dst[3 * kWorkers] = a3;
        }
    }
}

// rank16[w]: the block owner's rows in the range before word w; counts[b]: set bits of block b; partial[o]: owner o's
// rows in the range. b0 is a multiple of num_devices, so block b's owner is b % num_devices.
template <uint32_t num_devices, uint32_t words_per_block>
inline void range_ranks(
    const uint32_t* bits,
    uint16_t* rank16,
    uint32_t* counts,
    uint32_t* partial,
    uint32_t b0,
    uint32_t b1,
    uint32_t worker) {
    for (uint32_t o = worker; o < num_devices; o += kWorkers) {
        uint32_t n = 0;
        for (uint32_t b = b0 + o; b < b1; b += num_devices) {
            const uint32_t before = n;
#pragma GCC unroll 8
            for (uint32_t k = 0; k < words_per_block; ++k) {
                const uint32_t w = b * words_per_block + k;
                rank16[w] = static_cast<uint16_t>(n);
                n += __builtin_popcount(bits[w]);
            }
            counts[b] = n - before;
        }
        partial[o] = n;
    }
}

// Partials become exclusive prefixes over the ranges, in place.
template <uint32_t num_devices>
inline void owner_prefixes(uint32_t* partials, uint32_t ranges, uint32_t* totals, uint32_t worker) {
    for (uint32_t o = worker; o < num_devices; o += kWorkers) {
        uint32_t acc = 0;
        for (uint32_t r = 0; r < ranges; ++r) {
            const uint32_t t = partials[r * num_devices + o];
            partials[r * num_devices + o] = acc;
            acc += t;
        }
        totals[o] = acc;
    }
}

template <uint32_t num_devices, uint32_t unit>
inline void owner_bases(const uint32_t* totals, uint32_t* base, uint32_t worker) {
    for (uint32_t o = worker; o < num_devices; o += kWorkers) {
        uint32_t b = 0;
        for (uint32_t q = 0; q < o; ++q) {
            b += (totals[q] + unit - 1) / unit * unit;
        }
        base[o] = b;
    }
}

template <uint32_t num_devices, uint32_t stripe_chunk, uint32_t range_blocks, uint32_t pages_per_packet, uint32_t banks>
inline void rewrite(
    uint32_t* __restrict__ idx,
    uint32_t begin,
    uint32_t end,
    const uint32_t* __restrict__ bits,
    const uint16_t* __restrict__ rank16,
    const uint32_t* __restrict__ prefix,
    const uint32_t* __restrict__ base) {
    constexpr uint32_t unit = pages_per_packet * banks;
#pragma GCC unroll 2
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t v = idx[i];
        if (v == kSentinel) {
            continue;
        }
        const uint32_t blk = v / stripe_chunk;
        const uint32_t w = v >> 5;
        const uint32_t o = blk % num_devices;
        const uint32_t rank = prefix[(blk / range_blocks) * num_devices + o] + rank16[w] +
                              __builtin_popcount(bits[w] & ((1u << (v & 31)) - 1u));
        const uint32_t j = rank / pages_per_packet, p = rank % pages_per_packet;
        idx[i] = base[o] + (j % banks) + (j / banks) * unit + p * banks;
    }
}

template <
    uint32_t num_devices,
    uint32_t stripe_chunk,
    uint32_t range_blocks,
    uint32_t pages_per_packet,
    uint32_t banks,
    typename DoneSem>
inline void bases_and_rewrite(
    uint32_t* idx,
    uint32_t n_idx,
    const uint32_t* bits,
    const uint16_t* rank16,
    uint32_t* partials,
    uint32_t* misc,
    uint32_t worker,
    DoneSem&& done_sem) {
    owner_prefixes<num_devices>(partials, misc[kMiscRanges], misc + kMiscTotals, worker);
    post(done_sem(worker), kTotalled);
    for (uint32_t w = 0; w < kWorkers; ++w) {
        wait_at_least(done_sem(w), kTotalled);
    }
    owner_bases<num_devices, pages_per_packet * banks>(misc + kMiscTotals, misc + kMiscBase, worker);
    post(done_sem(worker), kBased);
    for (uint32_t w = 0; w < kWorkers; ++w) {
        wait_at_least(done_sem(w), kBased);
    }
    rewrite<num_devices, stripe_chunk, range_blocks, pages_per_packet, banks>(
        idx, share_begin(n_idx, worker), share_begin(n_idx, worker + 1), bits, rank16, partials, misc + kMiscBase);
    post(done_sem(worker), kRewritten);
}

}  // namespace sparse_push_plan
