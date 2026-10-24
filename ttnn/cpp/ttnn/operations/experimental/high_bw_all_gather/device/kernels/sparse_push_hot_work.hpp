// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The L1 loops of sparse_push_hot, shared by its five workers on each core (BRISC 0, NCRISC 1, TRISCs 2..4; see
// sparse_push_hot.cpp). Nothing here touches the NOC.

#pragma once

#include <cstdint>

#include "sparse_push_plan_work.hpp"

namespace sparse_push_hot {

using sparse_push_plan::kSentinel;
using sparse_push_plan::kWorkers;

// Circular buffers (same ids on the host side).
enum Cb : uint32_t {
    kIdx = 0,      // this core's slice of the top-k rows
    kCompact = 1,  // its rewritten slots
    kTables = 2,   // every requester's bitmap words of this core's rank range, then this chip's own
    kHot = 3,      // hot rows: every row at least `threshold` chips need (this core's range: then only multicast ones)
    kRank16 = 4,   // their ranks within the range; then (at cold_rank16) those of the cold rows
    kRanges = 5,   // their per (range, owner, part) counts, then prefixes (ranges_layout); then the cold rows'
    kMisc = 6,
    kOut = 7,   // a table header, then the hot table header
    kCold = 8,  // this chip's cold rows: needed and not multicast
};

// Offset of the cold rows' ranks behind the hot rows' in kRank16 (elements), so the rewrite picks either with a
// constant stride.
constexpr uint32_t cold_rank16(uint32_t bitmap_words) { return (bitmap_words * 2 + 63) / 64 * 64 / 2; }

// kRanges (words): the hot rows' per (range, owner, part) counts, [range][owner * 2 + part], then their exclusive
// prefixes over the ranges, transposed: [owner * 2 + part][range], range_stride apart; then the same for the cold rows
// at `cold`.
struct RangesLayout {
    uint32_t range_stride, prefixes, cold;
};
constexpr RangesLayout ranges_layout(uint32_t max_ranges, uint32_t num_devices) {
    const uint32_t stride = (max_ranges + 3) / 4 * 4;
    const uint32_t counts = (max_ranges * 2 * num_devices + 15) / 16 * 16;
    return {stride, counts, counts + (2 * num_devices * stride + 15) / 16 * 16};
}
// Words of the misc buffer.
constexpr uint32_t kMiscActiveBlocks = 0;  // largest active block count of any requester's table
constexpr uint32_t kMiscRanges = 1;        // rank ranges (range_blocks blocks each) over the active blocks
constexpr uint32_t kMiscSplitBlock = 2;    // the chunk start's block: blocks from it on are part 1
constexpr uint32_t kMiscHotSlots = 3;      // H: slots of the hot region
constexpr uint32_t kMiscSplit = 16;        // chunk start row, read from DRAM (64 B aligned)
constexpr uint32_t kMiscGo = 20;           // the exchange's go value, multicast by core 0
constexpr uint32_t kMiscLimits = 32;       // per requester (and this chip last): its table's active bitmap words
constexpr uint32_t kMiscBases = 72;        // [owner * 2 + part]: hot sub-run base
constexpr uint32_t kMiscTotals = 136;      // [owner * 2 + part]: hot rows of the sub-run (before the cap)
constexpr uint32_t kMiscColdTotals = 200;  // [owner * 2 + part]: this chip's cold rows from the owner
constexpr uint32_t kMiscRunBases = 264;    // [owner]: this chip's run of rows from the owner (the plan's layout)
constexpr uint32_t kMiscColdBases = 304;   // [owner * 2 + part]: this chip's run of rows from the owner, as slots
constexpr uint32_t kMiscWords = 368;

// Semaphores, as in the plan op: arrive and go for the cross-core exchange, step, one done flag per worker.
constexpr uint32_t kSemArrive = 0, kSemGo = 1, kSemStep = 2, kSemDone = 3;
constexpr uint32_t kNumSems = kSemDone + kWorkers;

// Steps. Within the count and the cold steps the workers hand off directly (kCounted then kRanked, kMasked then
// kColdRanked); each step starts on BRISC's post.
constexpr uint32_t kStepCount = 1, kCounted = 1, kRanked = 2;
constexpr uint32_t kStepCold = 3, kMasked = 3, kColdRanked = 4;
constexpr uint32_t kStepRewrite = 5;

using sparse_push_plan::post;
using sparse_push_plan::share_begin;
using sparse_push_plan::wait_at_least;

// Bits set in at least `threshold` of the n <= 32 words at words[0], words[stride], ...: a carry-save adder tree
// (Harley-Seal) into count planes 1, 2, 4, 8, 16, 32, then a most-significant-first comparison with the threshold.
template <uint32_t n>
inline uint32_t at_least(const uint32_t* words, uint32_t stride, uint32_t threshold) {
    static_assert(n >= 1 && n <= 32, "at most 32 words");
    auto x = [&](uint32_t i) -> uint32_t { return i < n ? words[i * stride] : 0u; };
    auto csa = [](uint32_t& hi, uint32_t& lo, uint32_t a, uint32_t b, uint32_t c) {
        const uint32_t u = a ^ b;
        hi = (a & b) | (u & c);
        lo = u ^ c;
    };
    uint32_t ones = 0, twos = 0, fours = 0, eights = 0, sixteens = 0, thirtytwos = 0;
#pragma GCC unroll 2
    for (uint32_t i = 0; i < 32; i += 16) {
        if (i >= n) {
            break;
        }
        uint32_t twos_a, twos_b, fours_a, fours_b, eights_a, eights_b, sixteens_a;
        csa(twos_a, ones, ones, x(i + 0), x(i + 1));
        csa(twos_b, ones, ones, x(i + 2), x(i + 3));
        csa(fours_a, twos, twos, twos_a, twos_b);
        csa(twos_a, ones, ones, x(i + 4), x(i + 5));
        csa(twos_b, ones, ones, x(i + 6), x(i + 7));
        csa(fours_b, twos, twos, twos_a, twos_b);
        csa(eights_a, fours, fours, fours_a, fours_b);
        csa(twos_a, ones, ones, x(i + 8), x(i + 9));
        csa(twos_b, ones, ones, x(i + 10), x(i + 11));
        csa(fours_a, twos, twos, twos_a, twos_b);
        csa(twos_a, ones, ones, x(i + 12), x(i + 13));
        csa(twos_b, ones, ones, x(i + 14), x(i + 15));
        csa(fours_b, twos, twos, twos_a, twos_b);
        csa(eights_b, fours, fours, fours_a, fours_b);
        csa(sixteens_a, eights, eights, eights_a, eights_b);
        thirtytwos |= sixteens & sixteens_a;
        sixteens ^= sixteens_a;
    }
    const uint32_t planes[6] = {ones, twos, fours, eights, sixteens, thirtytwos};
    uint32_t gt = 0, eq = ~0u;
#pragma GCC unroll 6
    for (int k = 5; k >= 0; --k) {
        if ((threshold >> k) & 1u) {
            eq &= planes[k];
        } else {
            gt |= eq & planes[k];
            eq &= ~planes[k];
        }
    }
    return gt | eq;
}

// Hot words [begin, end) of a rank range whose first bitmap word is w0: scratch holds every requester's words of
// the range (requester c's at scratch + c * range_words, zero past its active blocks).
template <uint32_t num_devices>
inline void count_hot(
    const uint32_t* scratch,
    uint32_t range_words,
    uint32_t w0,
    uint32_t begin,
    uint32_t end,
    uint32_t threshold,
    uint32_t* hot) {
    for (uint32_t k = begin; k < end; ++k) {
        hot[w0 + k] = at_least<num_devices>(scratch + k, range_words, threshold);
    }
}

// Ranks of the rows of `bits` in block range [b0, b1) (b0 a multiple of num_devices), for the owners
// o % kWorkers == worker: rank16[w] = rows of the block's (owner, part) in the range's earlier blocks plus bits before
// word w in its block; partial[o * 2 + part] = rows of (owner, part) in the range. Part 1 = blocks from the split
// block on.
template <uint32_t num_devices, uint32_t words_per_block>
inline void part_ranks(
    const uint32_t* bits,
    uint16_t* rank16,
    uint32_t* partial,
    uint32_t b0,
    uint32_t b1,
    uint32_t split_block,
    uint32_t worker) {
    for (uint32_t o = worker; o < num_devices; o += kWorkers) {
        uint32_t n[2] = {0, 0};
        for (uint32_t b = b0 + o; b < b1; b += num_devices) {
            const uint32_t part = b >= split_block ? 1 : 0;
            for (uint32_t k = 0; k < words_per_block; ++k) {
                const uint32_t w = b * words_per_block + k;
                rank16[w] = static_cast<uint16_t>(n[part]);
                n[part] += __builtin_popcount(bits[w]);
            }
        }
        partial[o * 2] = n[0];
        partial[o * 2 + 1] = n[1];
    }
}

// Owner o's exclusive prefixes over the ranges of its counts (both parts), into the transposed table, and their
// totals into totals[o * 2 + part]. With current_first, part 0's prefixes start at part 1's total (the cold rows'
// runs hold the current part first).
template <uint32_t num_devices, uint32_t range_stride, bool current_first>
inline void owner_prefixes(const uint32_t* counts, uint32_t ranges, uint32_t o, uint32_t* prefixes, uint32_t* totals) {
    constexpr uint32_t columns = 2 * num_devices;
    uint32_t acc1 = 0;
    for (uint32_t r = 0; r < ranges; ++r) {
        prefixes[(o * 2 + 1) * range_stride + r] = acc1;
        acc1 += counts[r * columns + o * 2 + 1];
    }
    const uint32_t first = current_first ? acc1 : 0;
    uint32_t acc0 = first;
    for (uint32_t r = 0; r < ranges; ++r) {
        prefixes[o * 2 * range_stride + r] = acc0;
        acc0 += counts[r * columns + o * 2];
    }
    totals[o * 2] = acc0 - first;
    totals[o * 2 + 1] = acc1;
}

// Words [w0 + begin, w0 + end) of one range, once the hot prefixes are in: each hot word loses the bits whose rank
// is past their sub-run's cap (in place: the multicast rows), and cold[w] = this chip's needed rows that are not
// (own = this chip's bitmap words of the range, zero past its active blocks).
template <
    uint32_t num_devices,
    uint32_t words_per_block,
    uint32_t range_blocks,
    uint32_t range_stride,
    uint32_t prefix_cap,
    uint32_t current_cap>
inline void mask_range(
    uint32_t* hot,
    const uint16_t* rank16,
    const uint32_t* prefixes,
    const uint32_t* own,
    uint32_t w0,
    uint32_t begin,
    uint32_t end,
    uint32_t split_block,
    uint32_t* cold) {
    for (uint32_t k = begin; k < end; ++k) {
        const uint32_t w = w0 + k;
        uint32_t x = hot[w];
        const uint32_t blk = w / words_per_block;
        const uint32_t part = blk >= split_block ? 1 : 0;
        const uint32_t cap = part ? current_cap : prefix_cap;
        const uint32_t first = prefixes[(blk % num_devices * 2 + part) * range_stride + blk / range_blocks] + rank16[w];
        if (first >= cap) {
            x = 0;
        } else if (first + static_cast<uint32_t>(__builtin_popcount(x)) > cap) {
            // Keep the lowest (earliest-row) cap - first set bits.
            uint32_t kept = 0;
            for (uint32_t keep = cap - first, m = x; keep != 0; --keep) {
                const uint32_t low = m & (0u - m);
                kept |= low;
                m ^= low;
            }
            x = kept;
        }
        hot[w] = x;
        cold[w] = own[k] & ~x;
    }
}

// Rewrite indices [begin, end) to compact slots. Every index is a row this chip needs, so it is cold (a bit of
// `cold`) or else multicast: then its slot is its hot sub-run's base plus the bank lane of its rank among the hot rows
// (`hot` may still hold the uncapped hot bits outside this core's range: the capped ones are the highest ranks, so the
// ranks of the kept ones are the same). A cold row's slot is its run's base (the run of rows from its owner, behind
// the hot region) plus the lane of its rank in the run, which holds the current part's cold rows first. rank16 and
// prefixes (transposed, see ranges_layout) hold the hot rows' tables, then the cold rows' at rank_cold and
// prefix_cold.
template <
    uint32_t num_devices,
    uint32_t stripe_chunk,
    uint32_t range_blocks,
    uint32_t pages_per_packet,
    uint32_t banks,
    uint32_t range_stride,
    uint32_t rank_cold,
    uint32_t prefix_cold>
inline void rewrite_hot(
    const uint32_t* __restrict__ natural,
    uint32_t* __restrict__ compact,
    uint32_t begin,
    uint32_t end,
    const uint32_t* __restrict__ hot,
    const uint32_t* __restrict__ cold,
    const uint16_t* __restrict__ rank16,
    const uint32_t* __restrict__ prefixes,
    const uint32_t* __restrict__ misc,
    uint32_t split_block) {
    constexpr uint32_t unit = pages_per_packet * banks;
    constexpr uint32_t bases_cold = kMiscColdBases - kMiscBases;
    const uint32_t* bases = misc + kMiscBases;
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t v = natural[i];
        if (v == kSentinel) {
            compact[i] = kSentinel;
            continue;
        }
        const uint32_t w = v >> 5, bit = v & 31;
        const uint32_t c = cold[w], h = hot[w];
        const uint32_t sel = (c >> bit) & 1u;
        const uint32_t word = sel ? c : h;
        const uint32_t blk = v / stripe_chunk;
        const uint32_t part = blk >= split_block ? 1 : 0;
        const uint32_t sub = (blk % num_devices) * 2 + part;
        const uint32_t rank = prefixes[sel * prefix_cold + sub * range_stride + blk / range_blocks] +
                              rank16[sel * rank_cold + w] + __builtin_popcount(word & ((1u << bit) - 1u));
        const uint32_t j = rank / pages_per_packet, p = rank % pages_per_packet;
        compact[i] = bases[sel * bases_cold + sub] + (j % banks) + (j / banks) * unit + p * banks;
    }
}

}  // namespace sparse_push_hot
