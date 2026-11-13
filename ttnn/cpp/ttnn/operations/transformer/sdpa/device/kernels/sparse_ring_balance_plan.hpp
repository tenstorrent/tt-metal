// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Load balancing of sparse_ring_sdpa (kept-only route), decided on device every call by the balance kernel
// (sparse_ring_balance.cpp): the static split of a device's (head, Q block) units over its SDPA cores, the
// cost-balanced unit table of every core and the cross-device row-block swaps, plus the words the balance kernel, the
// SDPA kernels, the gather writers and the program factory share. Plain integer code: every device computes the same
// result from the same costs.

#pragma once

#include <cstdint>

namespace sparse_balance {

// A core's unit table, written by the balance kernel into the core's cb_balance page: [count, entries], an entry a
// flat unit (< 2^15), with kImported when its row block is swapped in from another device. Ascending, so a head's
// units are consecutive.
constexpr uint32_t kTableWords = 64;
constexpr uint32_t kMaxCoreUnits = kTableWords - 1;
constexpr uint32_t kImported = 1u << 15;
constexpr uint32_t kUnitMask = kImported - 1;

// A unit's cost: its kept blocks holding a real token plus kUnitCost, the fixed per-unit work in kept blocks (plan and
// mask reads, step membership, normalization). Capped so a head's cost stays below 2^22.
constexpr uint32_t kUnitCost = 24;
constexpr uint32_t kMaxUnitCost = 4095;
constexpr uint32_t kMaxUnitsPerHead = 1024;

// Cost words per core in a table page (a 64-byte multiple: each core writes one aligned slice).
inline uint32_t cost_words_per_core(uint32_t max_units_per_core) { return (max_units_per_core + 15) / 16 * 16; }

// The balance kernel's L1 scratch, in words, each region 64-byte aligned: the cost pages of every device, the
// row-block costs, loads and partners of every device, the sort order (halfwords), heads / tails, moved flags (bytes),
// this device's partner page, then this device's unit costs, the cores' unit tables and loads, a head's bins (bounds,
// loads, counts), and every unit's place in its head's cost order and bin (halfwords).
struct Scratch {
    uint32_t tables, row_costs, loads, partners, order, heads, tails, moved, partner_page, unit_costs, unit_tables,
        core_loads, bins, unit_order, bin_of, words;
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
        unit_costs = take(units);
        unit_tables = take(cores * kTableWords);
        core_loads = take(2 * cores);
        bins = take(4 * cores);
        unit_order = take((units + 1) / 2);
        bin_of = take((units + 1) / 2);
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

// order = 0..n-1 sorted by costs, heaviest first, ties by index (Shell sort, Ciura's gaps).
inline void sort_heaviest_first(const uint32_t* costs, uint32_t n, uint16_t* order) {
    constexpr uint32_t gaps[] = {701, 301, 132, 57, 23, 10, 4, 1};
    for (uint32_t b = 0; b < n; ++b) {
        order[b] = static_cast<uint16_t>(b);
    }
    const auto before = [&](uint16_t x, uint16_t y) { return costs[x] > costs[y] || (costs[x] == costs[y] && x < y); };
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
        sort_heaviest_first(row_costs + r * q_blocks, swappable, order + r * swappable);
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

// Within a device: each head's units over the cores holding it in the static split, a core being a bin.
// improve_balance: up to kImproveRounds times, the busiest bin trades units with (or moves one to) the least loaded bin
// that admits a gain (bins tried from the least loaded up): the pair of units whose cost difference d is closest to
// half the bins' gap, which lowers the pair's maximum by min(d, gap - d). Bins keep between lo[b] and hi[b] units.
// order: the units, heaviest first. Deterministic: ties to the lower bin and unit.
constexpr uint32_t kImproveRounds = 8;

inline void improve_balance(
    const uint32_t* costs,
    uint32_t n,
    const uint16_t* order,
    uint32_t n_bins,
    const uint32_t* lo,
    const uint32_t* hi,
    uint32_t* load,
    uint32_t* count,
    uint16_t* bin_of) {
    for (uint32_t round = 0; round < kImproveRounds; ++round) {
        uint32_t big = 0;
        for (uint32_t b = 1; b < n_bins; ++b) {
            big = load[b] > load[big] ? b : big;
        }
        uint32_t best_u = n;
        uint32_t best_v = n;  // n: a move
        uint32_t best_gain = 0;
        uint32_t small = n_bins;
        for (uint32_t tried = 0; tried + 1 < n_bins && best_gain == 0; ++tried) {
            // The next bin by (load, index) after the last one tried.
            uint32_t next = n_bins;
            for (uint32_t b = 0; b < n_bins; ++b) {
                const bool after = small == n_bins || load[b] > load[small] || (load[b] == load[small] && b > small);
                if (b != big && after && (next == n_bins || load[b] < load[next])) {
                    next = b;
                }
            }
            if (next == n_bins || load[next] >= load[big]) {
                break;
            }
            small = next;
            const uint32_t gap = load[big] - load[small];
            const auto consider = [&](uint32_t u, uint32_t v) {
                const uint32_t cv = v < n ? costs[v] : 0u;
                if (costs[u] > cv && costs[u] - cv < gap) {
                    const uint32_t d = costs[u] - cv;
                    const uint32_t gain = d < gap - d ? d : gap - d;
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_u = u;
                        best_v = v;
                    }
                }
            };
            // Both bins' units in cost order: for each u of big, small's units just above and at or below
            // cost(u) - gap / 2.
            const bool can_move = count[big] > lo[big] && count[small] < hi[small];
            uint32_t j = 0;
            uint32_t above = n;
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t u = order[k];
                if (bin_of[u] != big) {
                    continue;
                }
                if (can_move) {
                    consider(u, n);
                }
                const uint32_t target = costs[u] > gap / 2 ? costs[u] - gap / 2 : 0u;
                for (; j < n && (bin_of[order[j]] != small || costs[order[j]] > target); ++j) {
                    above = bin_of[order[j]] == small ? order[j] : above;
                }
                if (above < n) {
                    consider(u, above);
                }
                if (j < n) {
                    consider(u, order[j]);
                }
            }
        }
        if (best_gain == 0) {
            break;
        }
        const uint32_t d = costs[best_u] - (best_v < n ? costs[best_v] : 0u);
        load[big] -= d;
        load[small] += d;
        bin_of[best_u] = static_cast<uint16_t>(small);
        if (best_v < n) {
            bin_of[best_v] = static_cast<uint16_t>(big);
        } else {
            --count[big];
            ++count[small];
        }
    }
}

// n units (costs[u] >= 1) over n_bins bins: bin b ends with between lo[b] and hi[b] units (sum lo <= n <= sum hi) and
// its load (in: what it holds already) grows by their costs; bin_of[u] = unit u's bin, order = the units heaviest
// first. Every unit, heaviest first, goes to the least loaded bin with a free slot (only to bins below lo once the
// rest are needed there), then improve_balance. Scratch: count (n_bins words).
inline void assign_balanced(
    const uint32_t* costs,
    uint32_t n,
    uint32_t n_bins,
    const uint32_t* lo,
    const uint32_t* hi,
    uint32_t* load,
    uint32_t* count,
    uint16_t* bin_of,
    uint16_t* order) {
    sort_heaviest_first(costs, n, order);
    uint32_t short_of = 0;  // units the bins below lo still need
    for (uint32_t b = 0; b < n_bins; ++b) {
        count[b] = 0;
        short_of += lo[b];
    }
    for (uint32_t k = 0; k < n; ++k) {
        const bool must_fill = n - k <= short_of;
        uint32_t best = n_bins;
        for (uint32_t b = 0; b < n_bins; ++b) {
            if (count[b] < hi[b] && (!must_fill || count[b] < lo[b]) && (best == n_bins || load[b] < load[best])) {
                best = b;
            }
        }
        short_of -= count[best] < lo[best] ? 1u : 0u;
        ++count[best];
        load[best] += costs[order[k]];
        bin_of[order[k]] = static_cast<uint16_t>(best);
    }
    improve_balance(costs, n, order, n_bins, lo, hi, load, count, bin_of);
}

// Every core's unit table (tables: cores x kTableWords) for this device's unit costs (costs[f], flat units): each
// head's units over the cores holding it in the static split, so head groups, chains and head boundaries stay as
// built. A core holding one head takes between `passes` (one unit per pass: a head group's members stream every pass
// in lockstep) and `cap` (<= kMaxCoreUnits) units; a core holding several (never a group member) its static units of
// each when they are dealt, at most `cap` in all when they are improved. The heads are assigned in order, then
// improved once more on the final loads of the cores spanning heads; the static split stays when it is no worse.
// Scratch: core_load (2 x cores words: loads, then units), bins (4 x cores words), order and bin_of (units halfwords
// each).
inline void plan_unit_tables(
    const StaticSplit& split,
    uint32_t passes,
    uint32_t cap,
    const uint32_t* costs,
    uint32_t* tables,
    uint32_t* core_load,
    uint32_t* bins,
    uint16_t* order,
    uint16_t* bin_of) {
    const uint32_t q_blocks = split.q_blocks;
    uint32_t* const lo = bins;
    uint32_t* const hi = bins + split.cores;
    uint32_t* const load = bins + 2 * split.cores;
    uint32_t* const count = bins + 3 * split.cores;
    uint32_t* const core_units = core_load + split.cores;
    for (uint32_t i = 0; i < split.cores; ++i) {
        core_load[i] = 0;
        core_units[i] = 0;
    }
    for (uint32_t sweep = 0; sweep < 2; ++sweep) {
        for (uint32_t f0 = 0; f0 < split.units; f0 += q_blocks) {
            const uint32_t first = split.core_of(f0);
            const uint32_t n_bins = split.core_of(f0 + q_blocks - 1) - first + 1;
            for (uint32_t b = 0; b < n_bins; ++b) {
                count[b] = 0;
            }
            for (uint32_t u = 0; sweep > 0 && u < q_blocks; ++u) {
                ++count[bin_of[f0 + u]];
            }
            for (uint32_t b = 0; b < n_bins; ++b) {
                const uint32_t i = first + b;
                core_units[i] -= count[b];  // its units of other heads
                // A core spanning heads: its static units of the head when dealing, any free slot when improving.
                const uint32_t a = split.start(i) > f0 ? split.start(i) : f0;
                const uint32_t e =
                    split.start(i) + split.count(i) < f0 + q_blocks ? split.start(i) + split.count(i) : f0 + q_blocks;
                const bool one_head = split.first_head(i) == split.last_head(i);
                lo[b] = one_head ? passes : 0u;
                hi[b] = one_head ? cap : (sweep == 0 ? e - a : cap - core_units[i]);
                load[b] = core_load[i];
            }
            if (sweep == 0) {
                assign_balanced(costs + f0, q_blocks, n_bins, lo, hi, load, count, bin_of + f0, order + f0);
            } else {
                improve_balance(costs + f0, q_blocks, order + f0, n_bins, lo, hi, load, count, bin_of + f0);
            }
            for (uint32_t b = 0; b < n_bins; ++b) {
                core_load[first + b] = load[b];
                core_units[first + b] += count[b];
            }
        }
    }
    uint32_t planned_max = 0;
    uint32_t static_max = 0;
    for (uint32_t i = 0, f = 0; i < split.cores; ++i) {
        planned_max = core_load[i] > planned_max ? core_load[i] : planned_max;
        uint32_t static_load = 0;
        for (uint32_t k = 0; k < split.count(i); ++k, ++f) {
            static_load += costs[f];
        }
        static_max = static_load > static_max ? static_load : static_max;
        tables[i * kTableWords] = 0;
    }
    const bool keep_static = static_max <= planned_max;
    for (uint32_t f = 0; f < split.units; ++f) {
        const uint32_t i = keep_static ? split.core_of(f) : split.core_of(f - f % q_blocks) + bin_of[f];
        uint32_t* const table = tables + i * kTableWords;
        table[1 + table[0]++] = f;
    }
}

}  // namespace sparse_balance
