// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Load balancing of sparse_ring_sdpa, decided on device every call by the balance kernel
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
// mask reads, step membership, normalization; the factory adds the pooled entries' work on the pooled route). Capped so
// a head's cost stays below 2^22.
constexpr uint32_t kUnitCost = 24;
constexpr uint32_t kMaxUnitCost = 4095;
constexpr uint32_t kMaxUnitsPerHead = 1024;

// Cost words per core in a table page (a 64-byte multiple: each core writes one aligned slice).
inline uint32_t cost_words_per_core(uint32_t max_units_per_core) { return (max_units_per_core + 15) / 16 * 16; }

// A device's table page: every core's cost slice (cost_words each), then the device's row-block costs (q_blocks words,
// padded to 64 bytes), which its balance kernel adds before the page goes to the other devices.
constexpr uint32_t table_page_words(uint32_t cores, uint32_t cost_words, uint32_t q_blocks) {
    return cores * cost_words + (q_blocks + 15) / 16 * 16;
}

// The balance kernel's L1 scratch, in words, each region 64-byte aligned: the table pages of every device, the
// row-block costs, loads and partners of every device, the sort order (halfwords), heads / tails, moved flags (bytes),
// this device's partner page, then this device's unit costs, the cores' unit tables and loads (2 x cores), and every
// unit's rank word (plan_unit_tables).
struct Scratch {
    uint32_t tables, row_costs, loads, partners, order, heads, tails, moved, partner_page, unit_costs, unit_tables,
        core_loads, ranked, words;
    Scratch(
        uint32_t devices, uint32_t cores, uint32_t cost_words, uint32_t q_blocks, uint32_t swappable, uint32_t units) {
        uint32_t w = 0;
        const auto take = [&](uint32_t n) {
            const uint32_t at = w;
            w += (n + 15) / 16 * 16;
            return at;
        };
        tables = take(devices * table_page_words(cores, cost_words, q_blocks));
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
        ranked = take(units);
        words = w;
    }
};

// The factory's static split: `units` flat units (head-major, `q_blocks` per head) over `cores` cores in order, base
// or base + 1 each (the first ones one more); with fewer units than cores the trailing cores hold none.
struct StaticSplit {
    uint32_t units = 0;
    uint32_t cores = 0;
    uint32_t q_blocks = 0;

    constexpr uint32_t base() const { return units / cores; }
    constexpr uint32_t extra() const { return units % cores; }
    constexpr uint32_t count(uint32_t i) const { return base() + (i < extra() ? 1u : 0u); }
    constexpr uint32_t start(uint32_t i) const { return i * base() + (i < extra() ? i : extra()); }
    // The core holding flat unit f (< units).
    constexpr uint32_t core_of(uint32_t f) const {
        const uint32_t long_units = extra() * (base() + 1);
        return f < long_units ? f / (base() + 1) : extra() + (f - long_units) / base();
    }
    constexpr uint32_t first_head(uint32_t i) const { return start(i) / q_blocks; }
    constexpr uint32_t last_head(uint32_t i) const { return (start(i) + count(i) - 1) / q_blocks; }
    // The most cores holding one head (a head's bins in plan_unit_tables).
    constexpr uint32_t max_bins() const {
        uint32_t most = 0;
        for (uint32_t f0 = 0; f0 < units; f0 += q_blocks) {
            const uint32_t n = core_of(f0 + q_blocks - 1) - core_of(f0) + 1;
            most = n > most ? n : most;
        }
        return most;
    }
};

// order[0..n) = 0..n-1 by keys (each <= top), heaviest first, ties by index: an LSD radix sort on top - key, 6 bits a
// pass (each pass stable). tmp: n halfwords.
template <typename Key>
inline void sort_heaviest_first(const Key* keys, uint32_t n, uint32_t top, uint16_t* order, uint16_t* tmp) {
    uint16_t* src = order;
    uint16_t* dst = tmp;
    for (uint32_t i = 0; i < n; ++i) {
        src[i] = static_cast<uint16_t>(i);
    }
    for (uint32_t shift = 0; shift < 32 && (shift == 0 || (top >> shift) != 0); shift += 6) {
        uint16_t at[65];  // at[d + 1]: entries of digit d, then where digit d's next entry goes
        for (uint32_t d = 0; d <= 64; ++d) {
            at[d] = 0;
        }
        for (uint32_t i = 0; i < n; ++i) {
            ++at[1 + (((top - keys[src[i]]) >> shift) & 63)];
        }
        for (uint32_t d = 1; d < 64; ++d) {
            at[d] += at[d - 1];
        }
        for (uint32_t i = 0; i < n; ++i) {
            dst[at[((top - keys[src[i]]) >> shift) & 63]++] = src[i];
        }
        uint16_t* const t = src;
        src = dst;
        dst = t;
    }
    for (uint32_t i = 0; src != order && i < n; ++i) {
        order[i] = src[i];
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

// order: each device's first `swappable` blocks heaviest first (sort_heaviest_first), devices x swappable halfwords.
// Returns the swaps made. Scratch: heads, tails (one word per device), moved (devices x q_blocks bytes).
inline uint32_t place_swaps(
    uint32_t devices,
    uint32_t q_blocks,
    uint32_t swappable,
    const uint32_t* row_costs,
    uint32_t* loads,
    uint32_t* partners,
    const uint16_t* order,
    uint32_t* heads,
    uint32_t* tails,
    uint8_t* moved) {
    uint64_t sum = 0;
    for (uint32_t r = 0; r < devices; ++r) {
        sum += loads[r];
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

// Within a device: each head's units over the cores holding it in the static split, a core being a bin. The planner
// works on one head at a time in HeadWork, which the balance kernel keeps in its RISC's local memory (L1 loads are
// uncached and several times slower): by rank (the head's units heaviest first, ties by index) the cost, the unit
// (index in the head) and the bin, and next (the next rank in the same bin, kEnd after the last); by bin its first
// rank, load, key (deal), unit count and the count's bounds lo / hi. Sized for q_blocks units and
// StaticSplit::max_bins() bins.
constexpr uint16_t kEnd = 0xFFFF;
struct HeadWork {
    uint16_t* cost;
    uint16_t* unit;
    uint16_t* next;
    uint8_t* bin;
    uint16_t* first;
    uint32_t* load;
    uint32_t* key;
    uint8_t* lo;
    uint8_t* hi;
    uint8_t* count;
};

// deal: every rank in turn (heaviest first) to the least loaded bin with a free slot (only to bins below lo once the
// rest are needed there); bin loads grow from what they hold already. sum lo <= n <= sum hi. key: a bin's load while
// it may take the next unit, else kClosed.
constexpr uint32_t kClosed = 0xFFFFFFFFu;

inline void deal(const HeadWork& w, uint32_t n, uint32_t n_bins) {
    uint32_t short_of = 0;  // units the bins below lo still need
    for (uint32_t b = 0; b < n_bins; ++b) {
        w.count[b] = 0;
        w.key[b] = w.hi[b] > 0 ? w.load[b] : kClosed;
        short_of += w.lo[b];
    }
    bool must_fill = false;
    for (uint32_t k = 0; k < n; ++k) {
        if (!must_fill && n - k <= short_of) {
            must_fill = true;  // from here on every unit goes to a bin below lo
            for (uint32_t b = 0; b < n_bins; ++b) {
                w.key[b] = w.count[b] < w.lo[b] ? w.load[b] : kClosed;
            }
        }
        uint32_t best = 0;
        uint32_t best_key = w.key[0];
        for (uint32_t b = 1; b < n_bins; ++b) {
            if (w.key[b] < best_key) {
                best = b;
                best_key = w.key[b];
            }
        }
        short_of -= w.count[best] < w.lo[best] ? 1u : 0u;
        ++w.count[best];
        w.load[best] += w.cost[k];
        w.bin[k] = static_cast<uint8_t>(best);
        const bool open = w.count[best] < (must_fill ? w.lo[best] : w.hi[best]);
        w.key[best] = open ? w.load[best] : kClosed;
    }
}

// improve: up to kImproveRounds times, the busiest bin trades units with (or moves one to) the least loaded bin that
// admits a gain (bins tried from the least loaded up): the pair of units whose cost difference d is closest to half the
// bins' gap, which lowers the pair's maximum by min(d, gap - d). Bins keep between lo[b] and hi[b] units. A pair is
// found in one merge-like walk of the two bins' rank lists, which a trade then re-splits. Deterministic: ties to the
// lower bin and rank.
constexpr uint32_t kImproveRounds = 8;

inline void improve(const HeadWork& w, uint32_t n, uint32_t n_bins) {
    const uint16_t* const cost = w.cost;
    uint16_t* const next = w.next;
    uint16_t* const first = w.first;
    const uint32_t* const load = w.load;
    for (uint32_t b = 0; b < n_bins; ++b) {
        first[b] = kEnd;
    }
    for (uint32_t k = n; k-- > 0;) {
        next[k] = first[w.bin[k]];
        first[w.bin[k]] = static_cast<uint16_t>(k);
    }
    for (uint32_t round = 0; round < kImproveRounds; ++round) {
        uint32_t big = 0;
        for (uint32_t b = 1; b < n_bins; ++b) {
            big = load[b] > load[big] ? b : big;
        }
        uint32_t best_u = kEnd;
        uint32_t best_v = kEnd;  // kEnd: a move
        uint32_t best_gain = 0;
        uint32_t small = n_bins;
        for (uint32_t tried = 0; tried + 1 < n_bins && best_gain == 0; ++tried) {
            // The next bin by (load, index) after the last one tried.
            uint32_t after_small = n_bins;
            for (uint32_t b = 0; b < n_bins; ++b) {
                const bool after = small == n_bins || load[b] > load[small] || (load[b] == load[small] && b > small);
                if (b != big && after && (after_small == n_bins || load[b] < load[after_small])) {
                    after_small = b;
                }
            }
            if (after_small == n_bins || load[after_small] >= load[big]) {
                break;
            }
            small = after_small;
            const uint32_t gap = load[big] - load[small];
            const auto consider = [&](uint32_t u, uint32_t v) {
                const uint32_t cv = v != kEnd ? cost[v] : 0u;
                if (cost[u] > cv && cost[u] - cv < gap) {
                    const uint32_t d = cost[u] - cv;
                    const uint32_t gain = d < gap - d ? d : gap - d;
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_u = u;
                        best_v = v;
                    }
                }
            };
            // For each u of big, small's units just above and at or below cost(u) - gap / 2.
            const bool can_move = w.count[big] > w.lo[big] && w.count[small] < w.hi[small];
            uint32_t j = first[small];
            uint32_t above = kEnd;
            for (uint32_t u = first[big]; u != kEnd; u = next[u]) {
                if (can_move) {
                    consider(u, kEnd);
                }
                const uint32_t target = cost[u] > gap / 2 ? cost[u] - gap / 2 : 0u;
                for (; j != kEnd && cost[j] > target; j = next[j]) {
                    above = j;
                }
                if (above != kEnd) {
                    consider(u, above);
                }
                if (j != kEnd) {
                    consider(u, j);
                }
            }
        }
        if (best_gain == 0) {
            break;
        }
        const uint32_t d = cost[best_u] - (best_v != kEnd ? cost[best_v] : 0u);
        w.load[big] -= d;
        w.load[small] += d;
        w.bin[best_u] = static_cast<uint8_t>(small);
        if (best_v != kEnd) {
            w.bin[best_v] = static_cast<uint8_t>(big);
        } else {
            --w.count[big];
            ++w.count[small];
        }
        // The two bins' ranks, merged in rank order, re-split by their bins now.
        uint32_t a = first[big];
        uint32_t c = first[small];
        uint32_t tails[2] = {kEnd, kEnd};  // big's, small's
        first[big] = kEnd;
        first[small] = kEnd;
        while (a != kEnd || c != kEnd) {
            uint32_t k;
            if (c == kEnd || (a != kEnd && a < c)) {
                k = a;
                a = next[a];
            } else {
                k = c;
                c = next[c];
            }
            const uint32_t b = w.bin[k];
            uint32_t& tail = tails[b == big ? 0 : 1];
            if (tail == kEnd) {
                first[b] = static_cast<uint16_t>(k);
            } else {
                next[tail] = static_cast<uint16_t>(k);
            }
            tail = k;
        }
        for (const uint32_t tail : tails) {
            if (tail != kEnd) {
                next[tail] = kEnd;
            }
        }
    }
}

// The static split is kept, without planning, when its busiest core is within 1 / kStaticSlack of the lower bound
// max(mean core load, largest unit). Random masks sit up to 16 % above it (a core holds whole units: 9 or 10 at 5 s),
// where a plan did not pay on device; the busiest device of a prefix, heavy-rows or causal mask is 24 % or more above.
constexpr uint32_t kStaticSlack = 5;

// A rank word (plan_unit_tables' `ranked`, a head's ranks between its two sweeps): cost << 18 | bin << 10 | unit.
constexpr uint32_t kRankUnitBits = 10;  // units per head <= kMaxUnitsPerHead
constexpr uint32_t kRankBinBits = 8;
constexpr uint32_t kMaxBins = 1u << kRankBinBits;

// Every core's unit table (tables: cores x kTableWords) for this device's unit costs (costs[f] <= kMaxUnitCost, flat
// units), entries flagged kImported where imported[f % q_blocks] (imported: q_blocks flags, or nullptr). Each head's
// units over the cores holding it in the static split, so head groups, chains and head boundaries stay as built. A
// core holding one head takes between `passes` (one unit per pass: a head group's members stream every pass in
// lockstep) and `cap` (<= kMaxCoreUnits) units; a core holding several (never a group member) its static units of each
// when they are dealt, at most `cap` in all when they are improved. The heads are dealt and improved in order, then
// improved once more on the final loads of the cores spanning heads. The static split stays when it is near the lower
// bound (kStaticSlack) or no worse than the plan. Scratch: core_load (2 x cores words: loads, then units), ranked
// (units words). Returns whether the plan is used.
inline bool plan_unit_tables(
    const StaticSplit& split,
    uint32_t passes,
    uint32_t cap,
    const uint32_t* costs,
    const uint8_t* imported,
    uint32_t* tables,
    uint32_t* core_load,
    uint32_t* ranked,
    const HeadWork& w) {
    const uint32_t q_blocks = split.q_blocks;
    const auto flag = [&](uint32_t b) { return imported != nullptr && imported[b] ? kImported : 0u; };
    // The static split's loads.
    uint32_t static_max = 0;
    uint32_t biggest = 0;
    uint64_t total = 0;
    for (uint32_t i = 0, f = 0; i < split.cores; ++i) {
        uint32_t load = 0;
        for (const uint32_t end = f + split.count(i); f < end; ++f) {
            const uint32_t c = costs[f];
            load += c;
            biggest = c > biggest ? c : biggest;
        }
        total += load;
        static_max = load > static_max ? load : static_max;
    }
    const auto write_static = [&]() {
        for (uint32_t i = 0; i < split.cores; ++i) {
            uint32_t* const table = tables + i * kTableWords;
            const uint32_t n = split.count(i);
            const uint32_t f0 = split.start(i);
            table[0] = n;
            for (uint32_t k = 0, b = n > 0 ? f0 % q_blocks : 0; k < n; ++k) {
                table[1 + k] = (f0 + k) | flag(b);
                b = b + 1 == q_blocks ? 0 : b + 1;
            }
        }
    };
    const uint32_t active = split.units < split.cores ? split.units : split.cores;
    const uint64_t mean = (total + active - 1) / active;
    const uint64_t bound = mean > biggest ? mean : biggest;
    if (static_cast<uint64_t>(kStaticSlack) * static_max <= (kStaticSlack + 1) * bound) {
        write_static();
        return false;
    }

    uint32_t* const core_units = core_load + split.cores;
    for (uint32_t i = 0; i < split.cores; ++i) {
        core_load[i] = 0;
        core_units[i] = 0;
        tables[i * kTableWords] = 0;
    }
    for (uint32_t sweep = 0; sweep < 2; ++sweep) {
        for (uint32_t f0 = 0; f0 < split.units; f0 += q_blocks) {
            const uint32_t first = split.core_of(f0);
            const uint32_t n_bins = split.core_of(f0 + q_blocks - 1) - first + 1;
            uint32_t* const head_ranked = ranked + f0;
            for (uint32_t b = 0; b < n_bins; ++b) {
                w.count[b] = 0;
            }
            if (sweep == 0) {
                // The head's costs by unit (in cost), its ranks, then the costs by rank.
                for (uint32_t u = 0; u < q_blocks; ++u) {
                    w.cost[u] = static_cast<uint16_t>(costs[f0 + u]);
                }
                sort_heaviest_first(w.cost, q_blocks, kMaxUnitCost, w.unit, w.next);
                for (uint32_t k = 0; k < q_blocks; ++k) {
                    w.next[k] = w.cost[w.unit[k]];
                }
                for (uint32_t k = 0; k < q_blocks; ++k) {
                    w.cost[k] = w.next[k];
                }
            } else {
                for (uint32_t k = 0; k < q_blocks; ++k) {
                    const uint32_t word = head_ranked[k];
                    w.unit[k] = static_cast<uint16_t>(word & ((1u << kRankUnitBits) - 1));
                    w.bin[k] = static_cast<uint8_t>(word >> kRankUnitBits);
                    w.cost[k] = static_cast<uint16_t>(word >> (kRankUnitBits + kRankBinBits));
                    ++w.count[w.bin[k]];
                }
            }
            for (uint32_t b = 0; b < n_bins; ++b) {
                const uint32_t i = first + b;
                core_units[i] -= w.count[b];  // its units of other heads (sweep 0: the count is 0)
                // A core spanning heads: its static units of the head when dealing, any free slot when improving.
                const uint32_t a = split.start(i) > f0 ? split.start(i) : f0;
                const uint32_t e =
                    split.start(i) + split.count(i) < f0 + q_blocks ? split.start(i) + split.count(i) : f0 + q_blocks;
                const bool one_head = split.first_head(i) == split.last_head(i);
                w.lo[b] = static_cast<uint8_t>(one_head ? passes : 0u);
                w.hi[b] = static_cast<uint8_t>(one_head ? cap : (sweep == 0 ? e - a : cap - core_units[i]));
                w.load[b] = core_load[i];
            }
            if (sweep == 0) {
                deal(w, q_blocks, n_bins);
            }
            improve(w, q_blocks, n_bins);
            for (uint32_t b = 0; b < n_bins; ++b) {
                core_load[first + b] = w.load[b];
                core_units[first + b] += w.count[b];
            }
            if (sweep == 0) {
                for (uint32_t k = 0; k < q_blocks; ++k) {
                    head_ranked[k] = w.unit[k] | static_cast<uint32_t>(w.bin[k]) << kRankUnitBits |
                                     static_cast<uint32_t>(w.cost[k]) << (kRankUnitBits + kRankBinBits);
                }
                continue;
            }
            // The head's units into its cores' tables, ascending: each unit's bin (in next), then the cores' entries
            // (first: each core's entries so far).
            for (uint32_t k = 0; k < q_blocks; ++k) {
                w.next[w.unit[k]] = w.bin[k];
            }
            for (uint32_t b = 0; b < n_bins; ++b) {
                w.first[b] = static_cast<uint16_t>(tables[(first + b) * kTableWords]);
            }
            for (uint32_t u = 0; u < q_blocks; ++u) {
                const uint32_t b = w.next[u];
                tables[(first + b) * kTableWords + 1 + w.first[b]++] = (f0 + u) | flag(u);
            }
            for (uint32_t b = 0; b < n_bins; ++b) {
                tables[(first + b) * kTableWords] = w.first[b];
            }
        }
    }
    uint32_t planned_max = 0;
    for (uint32_t i = 0; i < split.cores; ++i) {
        planned_max = core_load[i] > planned_max ? core_load[i] : planned_max;
    }
    if (static_max <= planned_max) {
        write_static();
        return false;
    }
    return true;
}

// The balance kernel's local-memory arrays (a static of the kernel, so the link checks they fit): the swap phase's
// row costs and their sort, then the planner's HeadWork; and this device's imported Q blocks throughout.
template <uint32_t QBlocks, uint32_t Bins>
struct LocalScratch {
    struct Swap {
        uint32_t row[QBlocks];
        uint16_t order[QBlocks];
        uint16_t tmp[QBlocks];
    };
    struct Plan {
        uint32_t load[Bins];
        uint32_t key[Bins];
        uint16_t cost[QBlocks];
        uint16_t unit[QBlocks];
        uint16_t next[QBlocks];
        uint16_t first[Bins];
        uint8_t bin[QBlocks];
        uint8_t lo[Bins];
        uint8_t hi[Bins];
        uint8_t count[Bins];
    };
    union {
        Swap swap;
        Plan plan;
    };
    uint8_t imported[QBlocks];

    HeadWork head() {
        return {
            plan.cost, plan.unit, plan.next, plan.bin, plan.first, plan.load, plan.key, plan.lo, plan.hi, plan.count};
    }
};

// An upper bound of sizeof(LocalScratch<q_blocks, bins>), and the most the balance kernel may take of its RISC's 8 KB
// local memory (the firmware's data and the stack hold the rest).
constexpr uint32_t local_scratch_bytes(uint32_t q_blocks, uint32_t bins) {
    const uint32_t swap = 8 * q_blocks + 8;
    const uint32_t plan = 8 * bins + 2 * (3 * q_blocks + bins) + q_blocks + 3 * bins + 16;
    return (swap > plan ? swap : plan) + q_blocks + 8;
}
constexpr uint32_t kMaxLocalScratchBytes = 3072;

}  // namespace sparse_balance
