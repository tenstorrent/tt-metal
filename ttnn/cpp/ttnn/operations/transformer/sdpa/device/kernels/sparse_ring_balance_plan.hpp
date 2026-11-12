// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Load balancing of sparse_ring_sdpa (kept-only route), decided on device every call by the balance kernel
// (sparse_ring_balance.cpp): the static split of a device's (head, Q block) units over its SDPA cores, the
// cost-balanced unit range of every core and the cross-device row-block swaps, plus the words the balance kernel, the
// SDPA kernels, the gather writers and the program factory share. Plain integer code: every device computes the same
// result from the same costs.

#pragma once

#include <cstdint>

namespace sparse_balance {

// A core's released unit range, written last into its range semaphore: valid bit, end << 16, start (< 2^15 units).
constexpr uint32_t kRangeValid = 1u << 31;
inline uint32_t encode_range(uint32_t start, uint32_t end) { return kRangeValid | (end << 16) | start; }
inline uint32_t range_start(uint32_t w) { return w & 0xFFFFu; }
inline uint32_t range_end(uint32_t w) { return (w >> 16) & 0x7FFFu; }

// A unit's cost: its kept blocks holding a real token plus kUnitCost, the fixed per-unit work in kept blocks (plan and
// mask reads, step membership, normalization). Capped so a head's cost stays below 2^22.
constexpr uint32_t kUnitCost = 24;
constexpr uint32_t kMaxUnitCost = 4095;
constexpr uint32_t kMaxUnitsPerHead = 1024;

// Cost words per core in a table page (a 64-byte multiple: each core writes one aligned slice).
inline uint32_t cost_words_per_core(uint32_t max_units_per_core) { return (max_units_per_core + 15) / 16 * 16; }

// The balance kernel's L1 scratch, in words, each region 64-byte aligned: the cost pages of every device, the
// row-block costs, loads and partners of every device, the sort order (halfwords), heads / tails, moved flags (bytes),
// this device's partner page, the cost prefix and the range starts.
struct Scratch {
    uint32_t tables, row_costs, loads, partners, order, heads, tails, moved, partner_page, prefix, starts, words;
    Scratch(
        uint32_t devices, uint32_t cores, uint32_t cost_words, uint32_t q_blocks, uint32_t swappable, uint32_t units) {
        uint32_t w = 0;
        const auto take = [&](uint32_t n) {
            const uint32_t at = w;
            w += (n + 15) / 16 * 16;
            return at;
        };
        tables = take(devices * cores * cost_words);
        row_costs = take(devices * q_blocks);
        loads = take(devices);
        partners = take(devices * q_blocks);
        order = take((devices * swappable + 1) / 2);
        heads = take(devices);
        tails = take(devices);
        moved = take((devices * q_blocks + 3) / 4);
        partner_page = take(q_blocks);
        prefix = take(units + 1);
        starts = take(cores + 1);
        words = w;
    }
};

// The factory's static split: `units` flat units (head-major, `q_blocks` per head) over `cores` cores in order, base
// or base + 1 each (the first ones one more); with fewer units than cores the trailing cores hold none.
struct StaticSplit {
    uint32_t units = 0;
    uint32_t cores = 0;
    uint32_t q_blocks = 0;

    uint32_t base() const { return units / cores; }
    uint32_t extra() const { return units % cores; }
    uint32_t count(uint32_t i) const { return base() + (i < extra() ? 1u : 0u); }
    uint32_t start(uint32_t i) const { return i * base() + (i < extra() ? i : extra()); }
    // The core holding flat unit f (< units).
    uint32_t core_of(uint32_t f) const {
        const uint32_t long_units = extra() * (base() + 1);
        return f < long_units ? f / (base() + 1) : extra() + (f - long_units) / base();
    }
    uint32_t first_head(uint32_t i) const { return start(i) / q_blocks; }
    uint32_t last_head(uint32_t i) const { return (start(i) + count(i) - 1) / q_blocks; }
};

// The largest t in [t_min, t_max] whose units from flat unit pos cost at most `budget` (prefix: running costs), or
// t_min - 1 when t_min units already cost more.
inline uint32_t take_within(const uint32_t* prefix, uint32_t pos, uint32_t t_min, uint32_t t_max, uint32_t budget) {
    const auto fits = [&](uint32_t t) { return prefix[pos + t] - prefix[pos] <= budget; };
    if (!fits(t_min)) {
        return t_min - 1;
    }
    uint32_t a = t_min;  // fits
    uint32_t b = t_max;
    while (a < b) {
        const uint32_t m = a + (b - a + 1) / 2;
        if (fits(m)) {
            a = m;
        } else {
            b = m - 1;
        }
    }
    return a;
}

// The static split stays unless the cut lowers the largest core cost by at least 1/kRecutMargin. The cost model leaves
// out what moving a cut changes beyond unit counts (head segments per pass, step groups, group lockstep), so a small
// modeled gain is noise; masks worth balancing gain far more. A core's cost is below two heads' (< 2^23), so the
// products fit 32 bits.
constexpr uint32_t kRecutMargin = 16;
inline bool static_cut_kept(uint32_t static_max, uint32_t planned_max) {
    return kRecutMargin * planned_max > (kRecutMargin - 1) * static_max;
}

// Every core's unit range for unit costs >= 1, prefix[f] = the cost of flat units [0, f) (units + 1 words): each core
// keeps its static heads (so head groups, chains and head boundaries stay as built) and takes a contiguous run of
// them; the ranges tile the units in core order. A core holding one head takes between `passes` (one unit per pass:
// a head group's members stream every pass in lockstep) and `cap` units, a core holding several (never a group
// member) at most `cap` in all. The largest core cost is minimized: the smallest budget (from the perfect split up by
// 1/8, then four halvings of the last step) for which taking, core by core and head by head, as many units as fit
// the budget and the bounds (leaving the later cores theirs) covers every head. A core continuing into the next head
// takes the rest of the current one and carries its units and cost into the next. Sets starts[i] for every core,
// starts[cores] = units: core i's range is [starts[i], starts[i + 1]).
class RangePlanner {
public:
    RangePlanner(const StaticSplit& split, uint32_t passes, uint32_t cap, const uint32_t* prefix) :
        split_(split), passes_(passes), cap_(cap), prefix_(prefix) {}

    void plan(uint32_t* starts) const {
        const uint32_t active = split_.units < split_.cores ? split_.units : split_.cores;
        uint32_t t = prefix_[split_.units] / active;
        for (uint32_t f = 0; f < split_.units; ++f) {
            const uint32_t c = prefix_[f + 1] - prefix_[f];
            t = c > t ? c : t;
        }
        t += t / 32 + 1;
        uint32_t t_bad = 0;
        while (!pass(t, nullptr)) {
            t_bad = t;
            t += t / 8 + 1;
        }
        for (uint32_t step = 0; step < 4 && t_bad + 1 < t; ++step) {
            const uint32_t mid = t_bad + (t - t_bad) / 2;
            if (pass(mid, nullptr)) {
                t = mid;
            } else {
                t_bad = mid;
            }
        }
        pass(t, starts);
        for (uint32_t i = active; i <= split_.cores; ++i) {
            starts[i] = split_.units;
        }
        uint32_t planned_max = 0;
        uint32_t static_max = 0;
        for (uint32_t i = 0; i < active; ++i) {
            const uint32_t p = prefix_[starts[i + 1]] - prefix_[starts[i]];
            const uint32_t s = prefix_[split_.start(i) + split_.count(i)] - prefix_[split_.start(i)];
            planned_max = p > planned_max ? p : planned_max;
            static_max = s > static_max ? s : static_max;
        }
        if (static_cut_kept(static_max, planned_max)) {
            for (uint32_t i = 0; i < active; ++i) {
                starts[i] = split_.start(i);
            }
        }
    }

private:
    // Head bounds of core i's part: (lo, hi), given the units it already holds in earlier heads.
    void bounds(uint32_t i, uint32_t held, uint32_t& lo, uint32_t& hi) const {
        const bool one_head = split_.first_head(i) == split_.last_head(i);
        lo = one_head ? passes_ : 0u;
        hi = cap_ - held;
    }

    // One greedy pass at budget t (starts == nullptr: only test it). False when the units do not fit.
    bool pass(uint32_t t, uint32_t* starts) const {
        const uint32_t q_blocks = split_.q_blocks;
        const uint32_t heads = split_.units / q_blocks;
        uint32_t pos = 0;
        uint32_t carry_core = split_.cores;  // the core continuing from the previous head, with its units and cost
        uint32_t carry_n = 0;
        uint32_t carry_c = 0;
        for (uint32_t h = 0; h < heads; ++h) {
            const uint32_t head_end = (h + 1) * q_blocks;
            const uint32_t first = split_.core_of(h * q_blocks);
            const uint32_t last = split_.core_of(head_end - 1);
            uint32_t lo_after = 0;
            uint32_t hi_after = 0;
            for (uint32_t i = first; i <= last; ++i) {
                uint32_t lo = 0;
                uint32_t hi = 0;
                bounds(i, i == carry_core ? carry_n : 0u, lo, hi);
                lo_after += lo;
                hi_after += hi;
            }
            for (uint32_t i = first; i <= last; ++i) {
                const uint32_t held_n = i == carry_core ? carry_n : 0u;
                const uint32_t held_c = i == carry_core ? carry_c : 0u;
                uint32_t lo = 0;
                uint32_t hi = 0;
                bounds(i, held_n, lo, hi);
                lo_after -= lo;
                hi_after -= hi;
                const uint32_t rest = head_end - pos;
                const bool continues = i == last && split_.last_head(i) > h;
                uint32_t take = rest;
                if (!continues) {
                    const uint32_t need = rest > hi_after ? rest - hi_after : 0u;  // the later cores cannot take more
                    const uint32_t t_min = lo > need ? lo : need;
                    const uint32_t t_max = hi < rest - lo_after ? hi : rest - lo_after;
                    if (t_min > t_max || held_c > t) {
                        return false;
                    }
                    take = take_within(prefix_, pos, t_min, t_max, t - held_c);
                    if (take + 1 == t_min) {
                        return false;
                    }
                } else if (take > hi || prefix_[pos + take] - prefix_[pos] + held_c > t) {
                    return false;
                }
                if (starts != nullptr && i != carry_core) {
                    starts[i] = pos;
                }
                if (continues) {
                    carry_n = held_n + take;
                    carry_c = held_c + prefix_[pos + take] - prefix_[pos];
                    carry_core = i;
                }
                pos += take;
            }
        }
        return true;
    }

    StaticSplit split_;
    uint32_t passes_;
    uint32_t cap_;
    const uint32_t* prefix_;
};

// Row-block swaps between devices. Every device has q_blocks row blocks (a Q block of every head); the first
// `swappable` may move (full blocks: a partial last block stays home). row_costs[r * q_blocks + b]: row block b of
// device r's cost; loads[r]: device r's whole cost. While the most loaded device is more than 1/32 above the mean, its
// heaviest unswapped block trades places with the least loaded device's lightest unswapped block, when that lowers the
// pair's maximum (the heavy side may skip to a lighter block that does, at most kScan ahead) by at least a quarter of
// an average block (a trade moves a whole row block's Q, mask rows and output). A slot trades at most once.
// partners[r * q_blocks + b] = kHome, or (r' << 16) | b': device r's slot b computes block b' of device r', and r''s
// slot b' computes r's block b. Deterministic: integer arithmetic, ties to the lower device / block.
constexpr uint32_t kHome = 0xFFFFFFFFu;
constexpr uint32_t kScan = 16;

inline void sort_blocks_by_cost(const uint32_t* row_costs, uint32_t n, uint16_t* order) {
    // Shell sort, heaviest first, ties by block index (Ciura's gaps).
    constexpr uint32_t gaps[] = {701, 301, 132, 57, 23, 10, 4, 1};
    for (uint32_t b = 0; b < n; ++b) {
        order[b] = static_cast<uint16_t>(b);
    }
    const auto before = [&](uint16_t x, uint16_t y) {
        return row_costs[x] > row_costs[y] || (row_costs[x] == row_costs[y] && x < y);
    };
    for (const uint32_t gap : gaps) {
        for (uint32_t i = gap; i < n; ++i) {
            const uint16_t v = order[i];
            uint32_t j = i;
            while (j >= gap && before(v, order[j - gap])) {
                order[j] = order[j - gap];
                j -= gap;
            }
            order[j] = v;
        }
    }
}

// Returns the swaps made. Scratch: order (devices x swappable halfwords), heads, tails (one word per device), moved
// (devices x q_blocks bytes).
inline uint32_t place_swaps(
    uint32_t devices,
    uint32_t q_blocks,
    uint32_t swappable,
    const uint32_t* row_costs,
    uint32_t* loads,
    uint32_t* partners,
    uint16_t* order,
    uint32_t* heads,
    uint32_t* tails,
    uint8_t* moved) {
    uint64_t sum = 0;
    for (uint32_t r = 0; r < devices; ++r) {
        sum += loads[r];
        sort_blocks_by_cost(row_costs + r * q_blocks, swappable, order + r * swappable);
        heads[r] = 0;
        tails[r] = swappable;
        for (uint32_t b = 0; b < q_blocks; ++b) {
            partners[r * q_blocks + b] = kHome;
            moved[r * q_blocks + b] = 0;
        }
    }
    uint32_t swaps = 0;
    for (uint32_t guard = 0; guard < devices * swappable; ++guard) {
        uint32_t hi = 0;
        uint32_t lo = 0;
        for (uint32_t r = 1; r < devices; ++r) {
            hi = loads[r] > loads[hi] ? r : hi;
            lo = loads[r] < loads[lo] ? r : lo;
        }
        if (hi == lo || 32ull * devices * loads[hi] <= 33ull * sum) {
            break;
        }
        const uint16_t* order_hi = order + hi * swappable;
        const uint16_t* order_lo = order + lo * swappable;
        while (heads[hi] < swappable && moved[hi * q_blocks + order_hi[heads[hi]]]) {
            ++heads[hi];
        }
        while (tails[lo] > 0 && moved[lo * q_blocks + order_lo[tails[lo] - 1]]) {
            --tails[lo];
        }
        if (heads[hi] == swappable || tails[lo] == 0) {
            break;
        }
        const uint32_t b_lo = order_lo[tails[lo] - 1];
        const uint32_t c_lo = row_costs[lo * q_blocks + b_lo];
        // The heaviest unswapped block of hi whose trade still lowers the pair's maximum.
        uint32_t b_hi = q_blocks;
        for (uint32_t j = heads[hi], seen = 0; j < swappable && seen < kScan; ++j) {
            const uint32_t b = order_hi[j];
            if (moved[hi * q_blocks + b]) {
                continue;
            }
            ++seen;
            const uint32_t c = row_costs[hi * q_blocks + b];
            if (c <= c_lo) {
                break;  // sorted: no lighter block of hi gains anything
            }
            if (loads[lo] + (c - c_lo) < loads[hi]) {
                b_hi = b;
                break;
            }
        }
        if (b_hi == q_blocks) {
            break;
        }
        const uint32_t gain = row_costs[hi * q_blocks + b_hi] - c_lo;
        if (4ull * gain * devices * q_blocks < sum) {
            break;
        }
        partners[hi * q_blocks + b_hi] = (lo << 16) | b_lo;
        partners[lo * q_blocks + b_lo] = (hi << 16) | b_hi;
        moved[hi * q_blocks + b_hi] = 1;
        moved[lo * q_blocks + b_lo] = 1;
        loads[hi] -= gain;
        loads[lo] += gain;
        ++swaps;
    }
    return swaps;
}

}  // namespace sparse_balance
