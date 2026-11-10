// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Dynamic head-aligned Q split, device allocator. The integer twin of the factory's
// sparse_ring_q_split_aligned("aligned:list:<w>", units, N, cores, max_q, min_q):
//   every unit (batch, head) starts with c0 = max(1, ceil(N / max_q)) cores; each spare core goes to the unit with the
//   largest ceil(N / c_u) x w_u (strictly larger wins: ties go to the lowest u), skipping units where one more core
//   would leave a core fewer than min_q blocks (min_q = the Q passes; 1 = one block per core is the finest split);
//   core j of unit u takes floor((j+1) N / c_u) - floor(j N / c_u) of its N Q blocks, in core order.
// The weights are integers (w_u = 4 x kept_u + 197 x N), so the factory's double products are exact and both pick the
// same cores. Pure integer code, no NoC: every core computes the same answer.
#include <cstdint>

namespace sparse_dyn_split {

constexpr uint32_t kMaxUnits = 64;

struct Entry {
    bool ok = false;          // false: the split does not exist (cores < units x c0, or cores left over)
    uint32_t unit = 0;        // (batch, head) unit of this core
    uint32_t q_start = 0;     // flat Q-block start: unit * N + the core's start within the unit
    uint32_t q_count = 0;     // Q blocks of this core
    uint32_t unit_core = 0;   // index of this core among its unit's cores (0 = the unit's first core)
    uint32_t unit_cores = 0;  // cores of its unit
    uint32_t first = 0;       // core index of the unit's first core (its cores are first .. first + unit_cores - 1)
};

// Q blocks of core j of a unit of n blocks over c cores.
// 32-bit on purpose: n (Q blocks per unit) and c (cores) are small (n * c < 2^32 by far), so the products are exact and
// the kernel needs no 64-bit division (__udivdi3, ~0.9 KB of reader text; kernel-config budget).
inline uint32_t unit_count(uint32_t n, uint32_t c, uint32_t j) { return (j + 1) * n / c - j * n / c; }

// Dense Q blocks (prefix spread, TT_SPARSE_RING / sparse_ring_sdpa dense_q_blocks): positions p (0..n) within every
// unit of this device whose Q block keeps every full key block (Sol's exact prefix rows), the same positions in every
// unit (the dense blocks are whole row blocks: all heads). The unit's cores then split it by cost, not by count:
// a dense position costs wd = 4 * n_full + 197, every other position the unit's mean routed cost; the weights use the
// head-weight units (4 per kept pair + 197 per Q block). n_dense = 0: the equal counts (unit_count), today's rule.
constexpr uint32_t kMaxDense = 8;
struct Dense {
    uint32_t n_dense = 0;
    uint32_t n_full = 0;     // kept pairs of a dense row: the full valid key blocks
    uint32_t cap = 0;        // most Q blocks per core in a weighted split (>= the allocator's max_q)
    uint32_t pos[kMaxDense] = {};
};

// One weighted split problem: n positions over c cores, each core [mn, cap] positions; routed positions weigh wn,
// the nd dense positions dp[] (ascending) weigh wd. All values stay far below 2^32 (n <= ~220, weights <= ~7,000).
struct Split {
    uint32_t n, c, cap, mn, nd, wn, wd;
    uint32_t dp[kMaxDense];
};

// The greedy at per-core budget T: each core in order takes as many positions as fit T (at most cap, leaving mn for
// every later core); routed runs between dense positions are taken in one step. Returns whether it covers the unit;
// with upto < c it stops at core `upto` and returns its range. Out of line: the binary search probes it ~20 times.
__attribute__((noinline)) inline bool greedy_split(
    const Split& s, uint32_t T, uint32_t upto, uint32_t& start, uint32_t& count) {
    uint32_t pos = 0;
    uint32_t di = 0;
    for (uint32_t k = 0; k < s.c; ++k) {
        const uint32_t left = s.c - 1 - k;
        const uint32_t avail = s.n - pos;
        if (avail < (left + 1) * s.mn) {
            return false;
        }
        uint32_t limit = avail - left * s.mn;
        limit = limit < s.cap ? limit : s.cap;
        uint32_t cnt = 0;
        uint32_t acc = 0;
        while (cnt < limit) {
            const uint32_t p = pos + cnt;
            const uint32_t nxt = di < s.nd ? s.dp[di] : s.n;
            if (p == nxt) {
                if (acc + s.wd > T) {
                    break;
                }
                acc += s.wd;
                ++cnt;
                ++di;
                continue;
            }
            uint32_t take = nxt - p;
            take = take < limit - cnt ? take : limit - cnt;
            const uint32_t room = T > acc ? (T - acc) / s.wn : 0;
            take = take < room ? take : room;
            acc += take * s.wn;
            cnt += take;
            if (take < nxt - p) {
                break;  // budget (or the core's limit) reached inside the routed run
            }
        }
        if (cnt < s.mn || (left == 0 && cnt != avail)) {
            return false;
        }
        if (k == upto) {
            start = pos;
            count = cnt;
            return true;
        }
        pos += cnt;
    }
    return pos == s.n;
}

// Core j's (start, count) within a unit of n blocks over c cores, its total weight w_unit = 4 * kept + 197 * n: the
// min-max contiguous split (the smallest budget T the greedy covers the unit with, by binary search, then the greedy
// at T). Per position: dense = 4 * n_full + 197, routed = the unit's routed mean. Every core of the unit runs the same
// code, so all agree. Falls back to unit_count when the weights cannot be dense-aware.
inline void weighted_range(
    uint32_t n, uint32_t c, uint32_t j, uint32_t w_unit, uint32_t min_q, const Dense& d, uint32_t& start,
    uint32_t& count) {
    Split s;
    s.n = n;
    s.c = c;
    s.cap = d.cap;
    s.mn = min_q > 0 ? min_q : 1;
    s.nd = 0;
    for (uint32_t i = 0; i < d.n_dense && i < kMaxDense; ++i) {
        const uint32_t v = d.pos[i];
        if (v < n) {
            uint32_t m = s.nd++;
            while (m > 0 && s.dp[m - 1] > v) {
                s.dp[m] = s.dp[m - 1];
                --m;
            }
            s.dp[m] = v;
        }
    }
    const uint32_t dense_kept4 = 4 * s.nd * d.n_full;
    if (s.nd == 0 || s.nd >= n || c < 2 || w_unit < 197 * n + dense_kept4) {
        start = j * n / c;
        count = unit_count(n, c, j);
        return;
    }
    s.wn = (w_unit - 197 * n - dense_kept4) / (n - s.nd) + 197;
    s.wd = 4 * d.n_full + 197;
    uint32_t lo = ((n - s.nd) * s.wn + s.nd * s.wd + c - 1) / c;  // the even share: no smaller budget fits
    lo = lo > s.wd ? lo : s.wd;
    uint32_t hi = s.nd * s.wd + s.cap * s.wn;  // feasible whenever n <= c * cap (the allocator guarantees it)
    hi = hi > lo ? hi : lo;
    uint32_t a = 0;
    uint32_t b = 0;
    if (!greedy_split(s, hi, c, a, b)) {
        start = j * n / c;
        count = unit_count(n, c, j);
        return;
    }
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (greedy_split(s, mid, c, a, b)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    greedy_split(s, lo, j, start, count);
}

// w[0..units): weights; c[0..units): scratch for the cores per unit (filled on return).
inline Entry entry(
    const uint32_t* w,
    uint16_t* c,
    uint32_t units,
    uint32_t n,
    uint32_t cores,
    uint32_t max_q,
    uint32_t min_q,
    uint32_t core,
    const Dense* dense = nullptr) {
    Entry e;
    if (units == 0 || units > kMaxUnits || n == 0 || cores < units) {
        return e;
    }
    const uint32_t mq = max_q > 0 ? max_q : 1;
    const uint32_t mn = min_q > 0 ? min_q : 1;
    const uint32_t c0 = (n + mq - 1) / mq > 1 ? (n + mq - 1) / mq : 1;
    if (c0 * units > cores || c0 * mn > n) {
        return e;
    }
    for (uint32_t u = 0; u < units; ++u) {
        c[u] = static_cast<uint16_t>(c0);
    }
    for (uint32_t spare = cores - c0 * units; spare > 0; --spare) {
        uint32_t best = units;
        uint64_t best_cost = 0;
        bool have = false;
        for (uint32_t u = 0; u < units; ++u) {
            if ((uint32_t(c[u]) + 1) * mn > n) {
                continue;
            }
            const uint64_t cost = static_cast<uint64_t>((n + c[u] - 1) / c[u]) * w[u];
            if (!have || cost > best_cost) {
                have = true;
                best_cost = cost;
                best = u;
            }
        }
        if (best == units) {
            return e;  // cores left over (every unit at its finest split): the factory fails the same way
        }
        ++c[best];
    }
    // Walk the units' cores to this core.
    uint32_t first = 0;
    for (uint32_t u = 0; u < units; ++u) {
        if (core < first + c[u]) {
            const uint32_t j = core - first;
            e.ok = true;
            e.unit = u;
            e.unit_core = j;
            e.unit_cores = c[u];
            e.first = first;
            if (dense != nullptr && dense->n_dense > 0) {
                uint32_t start = 0;
                uint32_t count = 0;
                weighted_range(n, c[u], j, w[u], mn, *dense, start, count);
                e.q_start = u * n + start;
                e.q_count = count;
                return e;
            }
            e.q_start = u * n + j * n / c[u];  // 32-bit: exact (see unit_count)
            e.q_count = unit_count(n, c[u], j);
            return e;
        }
        first += c[u];
    }
    return e;  // core beyond the placed cores (cannot happen when the cores add up)
}

}  // namespace sparse_dyn_split
