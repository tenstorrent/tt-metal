// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Runtime chunk-size picker, shared by the reader, writer, and compute kernels.
//
// The kernel reads this expert's actual token count device-side and picks
// chunk_M_tiles (hence per_core_M = chunk_M_tiles / GRID_Y and num_chunks)
// FROM THAT COUNT — the adaptive per_core_M work that used to be sized on the
// host from an `expected_tokens` argument now happens at runtime, so no caller
// has to know the token count in advance.
//
// CBs are sized on the host to the compile-time MAX shape (per_core_M = 8,
// chunk_M_tiles = 64; a big-model L1 guard may lower the max, passed in as
// `max_chunk`). The picker never returns more than `max_chunk`, so the runtime
// per_core_M always fits the allocated CBs; a smaller pick simply uses fewer of
// the reserved tiles and shrinks every M-dimension loop.
//
// The three kernels MUST derive identical chunk_M_tiles / per_core_M / chunk
// counts from the same count, or the reader/compute/writer disagree on the row
// mapping and FFN output lands on the wrong token rows. This is the single
// source of truth. Pure integer arithmetic (no NoC/CB/reg APIs) so the header
// is valid in BRISC, NCRISC, and TRISC translation units alike.
namespace adaptive_chunk {

// M-row cores; a chunk spans per_core_M * kGridY tile-rows. The host passes the grid height it
// launched (10 on Blackhole, 8 on smaller grids); group sizes are kGridY, kGridY / 2, kGridY / 4.
#ifndef UNIFIED_RE_GRID_Y
#define UNIFIED_RE_GRID_Y 8
#endif
constexpr uint32_t kGridY = UNIFIED_RE_GRID_Y;

// Chunk layout for `count_tiles` tile-rows, given the CB-sized maximum chunk
// `max_chunk` (= per_core_M_max * kGridY): a run of FULL chunks of max_chunk,
// then ONE tail chunk sized down to the remainder. This minimizes the chunk
// count (each chunk re-reads the full gate/up/down weights, the DRAM bottleneck)
// AND minimizes phantom M-work (the tail covers only the leftover rows), so
// per_core_M adapts to the actual token count per chunk with essentially no
// wasted rows — e.g. 160 tiles -> 64 + 64 + 32 (per_core_M 8,8,4), 3 chunks,
// zero phantom.
//
// CRITICAL — per_core_M BOUNDS WORK, IT DOES NOT SIZE CB BLOCKS.
//
// per_core_M is free to differ per expert and per chunk (that is the whole point:
// each expert gets the chunk shape its own token count deserves). It bounds MACs,
// tilize strips, DRAM reads, multicast payloads and emitted output rows.
//
// It must NOT size any circular-buffer block. cb_push_back / cb_pop_front wrap the
// FIFO pointer only when it lands EXACTLY on fifo_limit:
//
//     fifo_wr_ptr += num_words;
//     // this will basically reset fifo_wr_ptr to fifo_addr -- no other wrap is legal
//     ASSERT(fifo_wr_ptr <= fifo_limit);
//     if (fifo_wr_ptr == fifo_limit) { fifo_wr_ptr -= fifo_size; }
//                                              (tt_metal/hw/inc/api/dataflow/dataflow_api.h)
//
// A block size that changes between pushes leaves the pointer at an offset the
// next (larger) block does not divide; that push OVERSHOOTS fifo_limit, the
// equality fails, the wrap is skipped, the ASSERT is a no-op in release, and the
// CB pointer then runs away into neighbouring L1 for the rest of the kernel.
//
// Under the retired one-program-per-expert design this was invisible: every expert
// launched its own program, so the pointers restarted at fifo_addr and only the
// within-expert full-chunks-then-tail order mattered (and that order happens to
// realign). ONE program looping over all local experts carries the pointers over,
// so a per-expert block size corrupts L1 — e.g. on cb_activated (ring = 8 blocks at
// per_core_M=1, one push per chunk) a run of per_core_M=1 experts leaves the
// pointer at 3, and a following per_core_M=2 expert pushes 3 -> 5 -> 7 -> 9, which
// never equals 8.
//
// So the reader/compute kernels reserve/push/wait/pop a CONSTANT compile-time-max
// block on cb_x_rm, cb_in0_x, cb_gate_intermed, cb_activated and cb_mm_partials_*,
// and settle the runtime remainder with O(1) pointer-only bumps. This is the same
// pattern cb_in0_down_full and cb_out already use. The adaptive win is untouched:
// no MAC, tilize, DRAM read or multicast byte is spent on the padded rows.
//
// The tail per_core_M is therefore free to be ANY value in [1, per_core_M_max]:
// it bounds work only, and each CB still moves its whole constant block (the
// runtime remainder arrives as a pointer-only pad). The host asserts every ring
// is a whole number of the granule its kernels push, which is the property the
// FIFO wrap actually depends on.

// Clamp a DEVICE-PROVIDED token-tile count to what this expert can actually
// hold: its region (`m_tiles_full` tile-rows), and — as a backstop — the chunk
// loop the program was built for (`num_chunks_max` chunks of `max_chunk`).
//
// counts[] is produced on device (dispatch) and is never host-validated, so an
// over-capacity entry must not be allowed to drive the chunk loop past
// num_chunks_max: the CBs and the num_chunks compile-time arg are sized to that
// bound. ASSERT is a no-op unless watcher / lightweight kernel
// asserts are enabled, so the bound has to be enforced by arithmetic to hold in
// Release builds — the kernels ASSERT on top of it to still fail loudly in
// debug builds.
//
// Clamping the COUNT (not the chunk index) is what keeps the three kernels in
// lockstep: reader, compute and writer all derive effective_chunks, per_core_M
// and their row-validity guards from this same value, so they agree on the row
// mapping and the excess rows are uniformly dropped rather than emitted at the
// wrong offsets.
inline uint32_t clamp_count_tiles(
    uint32_t count_tiles, uint32_t max_chunk, uint32_t num_chunks_max, uint32_t m_tiles_full) {
    // Two independent caps; the tighter one wins.
    //   * m_tiles_full  - this expert's region is only this many tile-rows, so a
    //     larger count is bogus. Rows past it are dropped downstream anyway (the
    //     reader zero-fills past M_bound, the writer guards row < M_tiles_full), so
    //     without this cap an over-capacity count only bought wasted chunks.
    //   * num_chunks_max * max_chunk - the chunk-loop bound the CBs and the
    //     num_chunks compile-time arg were sized for.
    // m_tiles_full is normally much the tighter of the two: num_chunks_max is
    // ceil(m_tiles_full / min_chunk) with min_chunk <= max_chunk, so the loop cap
    // sits at roughly (max_chunk / min_chunk) * m_tiles_full. Both are kept so
    // neither bound can be violated if that host-side derivation ever changes.
    const uint32_t loop_cap = num_chunks_max * max_chunk;
    const uint32_t cap = (m_tiles_full < loop_cap) ? m_tiles_full : loop_cap;
    return (count_tiles < cap) ? count_tiles : cap;
}

// Active-token band, applied to the RAW token count before it becomes tiles.
//
// A hybrid dispatch runs this op and moe_fused_swiglu over the SAME counts vector and
// splits the experts by load: each op is given the band it is faster on and drops the
// rest. Dropping is spelled as count 0 because that is already the uniform skip every
// kernel agrees on -- effective_chunks becomes 0 and the expert costs no CB traffic, no
// collective and no semaphore. Applied to tokens, not tiles: a tile-granular bound
// cannot separate 300 from 320.
inline uint32_t count_in_band(uint32_t count_value, uint32_t min_tokens, uint32_t max_tokens) {
    return (count_value < min_tokens || count_value > max_tokens) ? 0u : count_value;
}

// Number of chunks for `count_tiles`: full chunks of max_chunk + one tail chunk.
inline uint32_t num_chunks(uint32_t count_tiles, uint32_t max_chunk) {
    if (count_tiles < 1) {
        return 0;
    }
    const uint32_t num_full = count_tiles / max_chunk;
    const uint32_t tail = count_tiles - num_full * max_chunk;
    return num_full + ((tail > 0) ? 1u : 0u);
}

// per_core_M for chunk index `c`: per_core_M_max for the full chunks; for the
// tail chunk, exactly the rows/core needed to cover the tail tiles, so no MAC,
// DRAM read or multicast byte is spent on a row the tail does not hold.
//
// `rows` is the number of M-row cores the expert is spread over: kGridY when the whole grid runs
// one expert at a time, fewer when the grid is split into expert-parallel row groups (below).
inline uint32_t per_core_M_for_chunk(uint32_t c, uint32_t count_tiles, uint32_t max_chunk, uint32_t rows = kGridY) {
    const uint32_t per_core_M_max = max_chunk / rows;
    const uint32_t num_full = count_tiles / max_chunk;
    if (c < num_full) {
        return per_core_M_max;
    }
    const uint32_t tail = count_tiles - num_full * max_chunk;  // > 0 for the tail chunk
    uint32_t need = (tail + rows - 1) / rows;                  // min rows/core to cover the tail
    if (need < 1) {
        need = 1;
    }
    return (need < per_core_M_max) ? need : per_core_M_max;
}
// ---------------------------------------------------------------------------------------------
// Expert-parallel row groups.
//
// Routed experts are small (a few tile-rows each in prefill), so spreading ONE expert over all
// kGridY M-rows leaves most rows computing padding and runs the experts strictly one after
// another. Instead the kGridY rows split into groups of consecutive M-rows; each group runs its
// own subset of experts, all groups concurrently. A group of `rows` rows covers an expert with
// per_core_M = ceil(count_tiles / rows), each group-column has its own weight multicast sender,
// and the x / activated multicasts are per M-row, so groups share nothing.
//
// The layout (how the rows split into groups) and the expert -> group map are picked at RUNTIME
// from the device-side counts. Every RISC of every core runs the identical integer schedule below
// over the same counts, so they agree on both without any host sync. An expert outside a core's
// group is skipped exactly like a count-0 expert (the uniform skip every kernel already agrees on).
constexpr uint32_t kMinGroupRows = 2;
constexpr uint32_t kMaxGroups = kGridY / kMinGroupRows;

// Group heights, top row first. The host builds one weight-multicast topology per layout, in
// table order, so the kernels index the runtime-arg sets by layout index.
//
// The first kNumUniformLayouts are the equal-height splits (kGridY, kGridY / 2, kGridY / 4 rows per
// group), scheduled exactly as before mixed layouts existed. The rest mix heights so that experts
// whose tile-row count does not divide a uniform height stop paying a phantom tile-row per core:
// a 9-tile expert runs per_core_M 3 on 3 rows instead of 5 on 2 or 2 on 5; a 3-tile expert runs
// per_core_M 1 on 3 rows instead of 2 on 2.
constexpr uint32_t kLayoutMaxGroups = 5;
struct GroupLayout {
    uint32_t groups;
    uint32_t rows[kLayoutMaxGroups];
};
constexpr uint32_t kNumUniformLayouts = 3;
constexpr GroupLayout kLayouts10[] = {
    {1, {10}}, {2, {5, 5}}, {5, {2, 2, 2, 2, 2}}, {4, {3, 3, 2, 2}}, {3, {4, 3, 3}}, {4, {4, 2, 2, 2}}};
constexpr GroupLayout kLayouts8[] = {{1, {8}}, {2, {4, 4}}, {4, {2, 2, 2, 2}}, {3, {3, 3, 2}}, {3, {4, 2, 2}}};
constexpr uint32_t num_layouts(uint32_t grid_y) {
    return grid_y == 10 ? sizeof(kLayouts10) / sizeof(kLayouts10[0]) : sizeof(kLayouts8) / sizeof(kLayouts8[0]);
}
constexpr const GroupLayout& layout_at(uint32_t grid_y, uint32_t layout) {
    return grid_y == 10 ? kLayouts10[layout] : kLayouts8[layout];
}

constexpr bool layouts_valid(uint32_t grid_y) {
    for (uint32_t li = 0; li < num_layouts(grid_y); ++li) {
        const GroupLayout& l = layout_at(grid_y, li);
        if (l.groups < 1 || l.groups > kLayoutMaxGroups || l.groups > grid_y / kMinGroupRows) {
            return false;
        }
        uint32_t sum = 0;
        for (uint32_t g = 0; g < l.groups; ++g) {
            if (l.rows[g] < kMinGroupRows) {
                return false;
            }
            sum += l.rows[g];
        }
        // Uniform layouts come first, in halving order, each with equal heights.
        if (li < kNumUniformLayouts) {
            const uint32_t rows = (li == 0) ? grid_y : (li == 1) ? grid_y / 2 : grid_y / 4;
            for (uint32_t g = 0; g < l.groups; ++g) {
                if (l.rows[g] != rows) {
                    return false;
                }
            }
        }
        if (sum != grid_y) {
            return false;
        }
    }
    return true;
}
static_assert(layouts_valid(10) && layouts_valid(8), "every layout must tile the grid with groups of >= 2 rows");
static_assert(kGridY == 10 || kGridY == 8, "row-group layouts exist for 8- and 10-row grids");
static_assert(kMaxGroups <= kLayoutMaxGroups, "layout table too narrow for this grid");

// The group holding grid row `gy`, with its first row and height.
inline uint32_t layout_group(const GroupLayout& l, uint32_t gy, uint32_t& base, uint32_t& rows) {
    uint32_t b = 0;
    for (uint32_t g = 0; g + 1 < l.groups; ++g) {
        if (gy < b + l.rows[g]) {
            base = b;
            rows = l.rows[g];
            return g;
        }
        b += l.rows[g];
    }
    base = b;
    rows = l.rows[l.groups - 1];
    return l.groups - 1;
}

// Weight-sender placement: grid column c's weight sender in the group of `rows` rows starting at
// grid row `base` sits on the group's TOP row when (c + base) is even and on its BOTTOM row
// otherwise, never on a middle one. So every column multicast can start at a rectangle corner and
// never wraps around the column of the NoC torus: the top sender's reader multicasts +y on NoC 0,
// the bottom sender's writer -y on NoC 1. On every 2-row group (all at even bases) this is the old
// `base + c % rows`. The host places the senders with this.
inline uint32_t sender_row_in_group(uint32_t c, uint32_t base, uint32_t rows) {
    return (((c + base) & 1u) == 0) ? 0u : rows - 1;
}
// The x-owner rotation below (which skips the columns c % rows == my_mt) in sender_row_in_group's terms:
// a corner row skips the columns of one parity -- (rows, my_mt) = (2, parity) -- and a middle row,
// which holds no sender, skips none -- (1, 0). Returned as rows | my_mt << 16.
inline uint32_t x_owner_key(uint32_t base, uint32_t rows, uint32_t my_mt) {
    if (my_mt != 0 && my_mt + 1 != rows) {
        return 1u;
    }
    return 2u | (((base + (my_mt == 0 ? 0u : 1u)) & 1u) << 16);
}

// Row-shared x tilize (row-major x): the grid column that reads, tilizes and multicasts gate/up
// K-block j of an M-row. That row is row `my_mt` of a `rows`-row group, whose weight sender for
// column c is the group's row c % rows (the host's placement). Columns whose core is this row's
// weight sender are skipped: its reader streams the column's weights, and x duty there (the stick
// reads under the same read barrier as the gate block, the tilize wait and the row multicast)
// would delay the column's weight multicast on every block. The remaining columns take the blocks
// in turn. Every core of the row shares (j, rows, my_mt), so all agree on the owner.
//
// Owner slots: the number of eligible columns (block j goes to slot j % slots).
inline uint32_t x_owner_slots(uint32_t grid_x, uint32_t rows, uint32_t my_mt) {
    if (rows < 2 || my_mt >= rows) {
        return grid_x;
    }
    const uint32_t senders = (my_mt < grid_x) ? (grid_x - 1 - my_mt) / rows + 1 : 0;
    return grid_x - senders;
}
// Slot of column `col` (its rank among the eligible columns), or `slots` if it is never an owner.
inline uint32_t x_owner_rank(uint32_t col, uint32_t rows, uint32_t my_mt, uint32_t slots) {
    if (rows < 2 || my_mt >= rows) {
        return col;
    }
    if (col % rows == my_mt) {
        return slots;
    }
    return col - ((col > my_mt) ? (col - 1 - my_mt) / rows + 1 : 0);
}
// Owner column of block j. Out of line: the reader calls it a few times per K-block.
inline __attribute__((noinline)) uint32_t x_owner_col(uint32_t j, uint32_t grid_x, uint32_t rows, uint32_t my_mt) {
    const uint32_t r = j % x_owner_slots(grid_x, rows, my_mt);
    if (rows < 2 || my_mt >= rows) {
        return r;
    }
    // r-th column c (ascending) with c % rows != my_mt: each run of `rows` columns holds rows - 1.
    const uint32_t per = rows - 1;
    const uint32_t q = r / per;
    const uint32_t rem = r - q * per;
    return q * rows + rem + (rem >= my_mt ? 1u : 0u);
}

// Modelled cost of one expert on a group of `rows` M-rows, in units of one per-core tile-row of
// work: the per_core_M summed over its chunks, plus one unit per chunk for the chunk's fixed
// pipeline fill/drain and its full re-read of the expert's weights. 0 for an expert with no rows.
inline uint32_t expert_cost(uint32_t count_tiles, uint32_t rows, uint32_t per_core_M_max) {
    const uint32_t max_chunk = per_core_M_max * rows;
    const uint32_t chunks = num_chunks(count_tiles, max_chunk);
    if (chunks == 0) {
        return 0;
    }
    uint32_t units = (count_tiles / max_chunk) * per_core_M_max;
    if (count_tiles % max_chunk != 0) {
        units += per_core_M_for_chunk(chunks - 1, count_tiles, max_chunk, rows);
    }
    return units + chunks;
}

// expert_cost without a divide for the usual single-chunk expert (these RISC-Vs have no hardware
// divider, and the schedule runs on every RISC before any work starts): ceil(tiles / rows) + 1 by
// at most per_core_M_max additions. Identical values; larger experts take the general path.
inline uint32_t group_cost(uint32_t count_tiles, uint32_t rows, uint32_t per_core_M_max) {
    if (count_tiles == 0) {
        return 0;
    }
    if (count_tiles > per_core_M_max * rows) {
        return expert_cost(count_tiles, rows, per_core_M_max);
    }
    uint32_t per_core_M = 1;
    for (uint32_t covered = rows; covered < count_tiles; covered += rows) {
        ++per_core_M;
    }
    return per_core_M + 1;
}

// The launch planner below runs once per kernel, before any work. On the compute kernel (UNPACK) it
// is built at -O2, as the dataflow kernels build it, not at the compute kernels' -O3: -O3 roughly
// doubles its code (~9.5 KB vs ~4.5 KB) for no measurable gain, and every RISC binary of a core has
// to fit the kernel config buffer together (r03-b01-a01 overflowed it). The reader's -O2 copy plans
// at the same speed, and compute waits for the reader's first weight multicast anyway.
// On the writer (BRISC) it is built at -Os, ~0.8 KB smaller than -O2, for the same kernel config buffer
// reason (the h7168 programs are at the limit). Only the column weight senders' writers need the plan
// early (their first `up` read), and that read is released by the reader's -O2 copy anyway
// (r04-b01-a02 measured first-GU-start unchanged).
#if defined(COMPILE_FOR_TRISC)
#pragma GCC push_options
#pragma GCC optimize("O2")
#elif defined(COMPILE_FOR_BRISC)
#pragma GCC push_options
#pragma GCC optimize("Os")
#endif
constexpr uint8_t kNoGroup = 0xFF;
constexpr uint32_t kNoSchedule = 0xFFFFFFFFu;

// Uniform layouts: list scheduling in expert order, each expert to the least-loaded group (lowest
// index on a tie). Returns the makespan; fills owner[] when given.
template <typename TilesOf>
inline uint32_t schedule_in_order(
    const GroupLayout& l, const TilesOf& tiles_of, uint32_t num_experts, uint32_t per_core_M_max, uint8_t* owner) {
    uint32_t loads[kLayoutMaxGroups] = {0};
    for (uint32_t e = 0; e < num_experts; ++e) {
        const uint32_t cost = group_cost(tiles_of(e), l.rows[0], per_core_M_max);
        uint32_t best = 0;
        if (cost > 0) {
            for (uint32_t g = 1; g < l.groups; ++g) {
                if (loads[g] < loads[best]) {
                    best = g;
                }
            }
            loads[best] += cost;
        }
        if (owner != nullptr) {
            owner[e] = (cost > 0) ? static_cast<uint8_t>(best) : kNoGroup;
        }
    }
    uint32_t makespan = 0;
    for (uint32_t g = 0; g < l.groups; ++g) {
        makespan = (loads[g] > makespan) ? loads[g] : makespan;
    }
    return makespan;
}

// Mixed layouts: largest expert first (ties by index), each to the group where it is cheapest
// among those it fits in under `cap` (ties -> lower load -> lower index). Returns the makespan, or
// kNoSchedule when some expert fits nowhere. Distinct tile counts are walked largest-first instead
// of sorting, so no per-expert buffer beyond owner[] is needed.
template <typename TilesOf>
inline uint32_t schedule_lpt(
    const GroupLayout& l,
    const TilesOf& tiles_of,
    uint32_t num_experts,
    uint32_t per_core_M_max,
    uint32_t cap,
    uint8_t* owner) {
    uint32_t loads[kLayoutMaxGroups] = {0};
    uint32_t cur = 0;
    for (uint32_t e = 0; e < num_experts; ++e) {
        const uint32_t t = tiles_of(e);
        cur = (t > cur) ? t : cur;
        if (owner != nullptr) {
            owner[e] = kNoGroup;
        }
    }
    while (cur > 0) {
        uint32_t cost[kLayoutMaxGroups];
        for (uint32_t g = 0; g < l.groups; ++g) {
            cost[g] = group_cost(cur, l.rows[g], per_core_M_max);
        }
        uint32_t next = 0;
        for (uint32_t e = 0; e < num_experts; ++e) {
            const uint32_t t = tiles_of(e);
            if (t != cur) {
                next = (t < cur && t > next) ? t : next;
                continue;
            }
            uint32_t best = kLayoutMaxGroups;
            for (uint32_t g = 0; g < l.groups; ++g) {
                if (loads[g] + cost[g] > cap) {
                    continue;
                }
                if (best == kLayoutMaxGroups || cost[g] < cost[best] ||
                    (cost[g] == cost[best] && loads[g] < loads[best])) {
                    best = g;
                }
            }
            if (best == kLayoutMaxGroups) {
                return kNoSchedule;
            }
            loads[best] += cost[best];
            if (owner != nullptr) {
                owner[e] = static_cast<uint8_t>(best);
            }
        }
        cur = next;
    }
    uint32_t makespan = 0;
    for (uint32_t g = 0; g < l.groups; ++g) {
        makespan = (loads[g] > makespan) ? loads[g] : makespan;
    }
    return makespan;
}

// The layout with the smallest modelled makespan, and owner[e] = the group (within that layout)
// that runs local expert e, kNoGroup for an expert with no rows. `tiles_of(e)` returns local expert
// e's (band-filtered, clamped) tile-row count.
//
// Uniform layouts are compared first, ties keeping the larger groups, so a single expert or large
// token counts stay on the whole-grid schedule. A mixed layout replaces the winner only when it is
// STRICTLY better: it is scheduled under a cap one below the best makespan so far.
//
// `uniform[li]` is schedule_in_order's makespan for uniform layout li (uniform_makespans), shared
// with make_plan so the uniform layouts are scheduled once per launch.
template <uint32_t NumExperts, typename TilesOf>
inline uint32_t plan_groups(
    const TilesOf& tiles_of,
    uint32_t per_core_M_max,
    const uint32_t (&uniform)[kNumUniformLayouts],
    uint8_t (&owner)[NumExperts]) {
    uint32_t best_layout = 0;
    uint32_t best = kNoSchedule;
    for (uint32_t li = 0; li < kNumUniformLayouts; ++li) {
        const uint32_t m = uniform[li];
        if (m < best) {
            best = m;
            best_layout = li;
        }
    }
    uint32_t best_cap = 0;
    for (uint32_t li = kNumUniformLayouts; li < num_layouts(kGridY); ++li) {
        if (best < 2) {
            break;  // no rows at all: nothing to improve on
        }
        const uint32_t cap = best - 1;
        const uint32_t m = schedule_lpt(layout_at(kGridY, li), tiles_of, NumExperts, per_core_M_max, cap, nullptr);
        if (m < best) {
            best = m;
            best_layout = li;
            best_cap = cap;
        }
    }
    if (best_layout < kNumUniformLayouts) {
        schedule_in_order(layout_at(kGridY, best_layout), tiles_of, NumExperts, per_core_M_max, owner);
    } else {
        schedule_lpt(layout_at(kGridY, best_layout), tiles_of, NumExperts, per_core_M_max, best_cap, owner);
    }
    return best_layout;
}

// ---------------------------------------------------------------------------------------------
// Two-phase schedule.
//
// One group size for the whole launch leaves a tail whenever the experts do not divide into the
// groups: 12 experts on 5 two-row groups run 3/3/2/2/2, so for the last third of the call two
// groups stream an expert each at per_core_M 2 (compute-bound) while three groups and half the
// DRAM bandwidth sit idle. The plan below instead runs the full waves on the picked group size
// (phase A) and then the LEFTOVER experts on larger groups (phase B), where per_core_M drops and
// the remaining streams are DRAM-bound again. No weight byte is read twice and every expert still
// runs whole on one group, so the numerics are unchanged.
//
// Phase A deals its experts round-robin, so every phase-A group runs the same number of experts,
// and the plan requires them all to take the same number of chunks. Every core has therefore
// pushed the same number of blocks into every ring when it enters phase B, which is what the
// phase-B weight multicast relies on: a sender writes into each receiver's ring at its OWN write
// pointer, and phase-B column groups span rows of different phase-A groups. A phase-B group
// needs no barrier: its multicasts wait on receiver acks, and phase A uses its own weight
// semaphore pair, so an early phase-B ack never reaches a core still sending for phase A.

// Membership and order. Phase A takes the n_a experts with the most tile-rows: every expert with
// more than `thresh`, then the first `quota_eq` (in expert order) with exactly `thresh`.
// rows_a == 0 means there is no phase A and phase B is today's single-phase schedule.
struct Plan {
    uint32_t rows_a = 0;
    uint32_t rows_b = kGridY;
    uint32_t thresh = 0;
    uint32_t quota_eq = 0;
    // Two-phase plans only: phase B's layout (uniform or mixed-height). Phase B runs each leftover
    // expert alone on one group, the k-th largest on group k (taller groups come first in a layout).
    uint32_t layout_b = 0;
};

// Index of the uniform layout with `rows` rows per group (the first kNumUniformLayouts table
// entries are kGridY, kGridY / 2, kGridY / 4 rows per group; layouts_valid checks that order).
inline uint32_t uniform_layout_index(uint32_t rows) { return (rows >= kGridY) ? 0u : (rows >= kGridY / 2) ? 1u : 2u; }

// List scheduling in expert order: each expert goes to the least-loaded group (lowest index on a
// tie), as schedule_in_order does for a uniform layout.
struct GroupScheduler {
    uint32_t groups = 1;
    uint32_t loads[kMaxGroups] = {0};

    void init(uint32_t num_groups) {
        groups = num_groups;
        for (uint32_t g = 0; g < kMaxGroups; ++g) {
            loads[g] = 0;
        }
    }
    // Charge `cost` (> 0) to the least-loaded group and return that group.
    uint32_t assign(uint32_t cost) {
        uint32_t best = 0;
        for (uint32_t g = 1; g < groups; ++g) {
            if (loads[g] < loads[best]) {
                best = g;
            }
        }
        loads[best] += cost;
        return best;
    }
};

// Uniform group size (rows per group) with the smallest in-order makespan; ties keep the larger
// group. The same pick plan_groups makes among its uniform layouts.
template <typename TilesOf>
inline void uniform_makespans(
    const TilesOf& tiles_of, uint32_t num_experts, uint32_t per_core_M_max, uint32_t (&uniform)[kNumUniformLayouts]) {
    for (uint32_t li = 0; li < kNumUniformLayouts; ++li) {
        uniform[li] = schedule_in_order(layout_at(kGridY, li), tiles_of, num_experts, per_core_M_max, nullptr);
    }
}
inline uint32_t pick_group_rows(const uint32_t (&uniform)[kNumUniformLayouts]) {
    uint32_t best_rows = kGridY;
    uint32_t best = kNoSchedule;
    for (uint32_t li = 0; li < kNumUniformLayouts; ++li) {
        if (uniform[li] < best) {
            best = uniform[li];
            best_rows = layout_at(kGridY, li).rows[0];
        }
    }
    return best_rows;
}

// Deals one phase's experts to that phase's groups. Feed it EVERY local expert's tile count, in
// expert order; it returns the group that runs the expert in this phase, or kNoGroup. Every kernel
// walks it identically, so all agree on the owner of every expert.
struct PhaseDealer {
    Plan plan;
    uint32_t phase = 0;  // 0 = phase A, 1 = phase B
    uint32_t rows = 0;   // this phase's group size; 0 = the phase is empty
    uint32_t groups = 0;
    uint32_t per_core_M_max = 1;
    uint32_t pos = 0;      // phase-A round-robin position
    uint32_t eq_seen = 0;  // experts with exactly plan.thresh tile-rows seen so far
    GroupScheduler sched;

    void begin(const Plan& p, uint32_t ph, uint32_t pm_max) {
        plan = p;
        phase = ph;
        rows = (ph == 0) ? p.rows_a : p.rows_b;
        groups = (rows > 0) ? kGridY / rows : 0;
        per_core_M_max = pm_max;
        pos = 0;
        eq_seen = 0;
        sched.init(groups > 0 ? groups : 1);
    }
    bool active() const { return rows > 0; }

    uint32_t deal(uint32_t tiles) {
        if (tiles == 0) {
            return kNoGroup;
        }
        bool in_a = false;
        if (plan.rows_a != 0) {
            if (tiles > plan.thresh) {
                in_a = true;
            } else if (tiles == plan.thresh) {
                in_a = eq_seen < plan.quota_eq;
                ++eq_seen;
            }
        }
        if (in_a != (phase == 0)) {
            return kNoGroup;
        }
        if (phase == 0) {
            return pos++ % groups;
        }
        return sched.assign(group_cost(tiles, rows, per_core_M_max));
    }
};

// Time model used to choose the plan, in hundredths of `c` = the time to stream one expert's
// weights at the full DRAM rate. An expert costs its per-core compute (kModelRow per tile-row of
// per_core_M, kModelChunk per chunk) or its share of DRAM while `streams` groups stream at once,
// whichever is larger. Calibrated on r01-b04-a04's per-group end times: the full waves run at the
// DRAM share, and a lone 2-row group runs a 4-tile expert in ~3.3 c (2 x kModelRow).
constexpr uint32_t kModelRow = 163;
constexpr uint32_t kModelChunk = 20;
constexpr uint32_t kModelStream = 100;
constexpr uint32_t kModelPhaseSwitch = 20;
// A two-phase plan must beat the single-phase one by this many percent.
constexpr uint32_t kModelMarginPct = 5;

inline uint32_t model_expert(uint32_t tiles, uint32_t rows, uint32_t per_core_M_max, uint32_t streams) {
    // group_cost == expert_cost, and a single-chunk expert needs no num_chunks divide.
    const uint32_t chunks =
        (tiles <= per_core_M_max * rows) ? ((tiles > 0) ? 1u : 0u) : num_chunks(tiles, per_core_M_max * rows);
    const uint32_t row_units = group_cost(tiles, rows, per_core_M_max) - chunks;  // sum of per_core_M
    const uint32_t compute = kModelRow * row_units + kModelChunk * chunks;
    const uint32_t dram = kModelStream * streams * chunks;
    return (compute > dram) ? compute : dram;
}

// Modelled makespan of one phase of `plan`. The i-th expert of a group overlaps with the i-th of
// every other group that has one, so it streams alongside that many groups.
template <typename TilesOf>
inline uint32_t model_phase(
    const TilesOf& tiles_of, uint32_t num_experts, uint32_t per_core_M_max, const Plan& plan, uint32_t phase) {
    PhaseDealer d;
    d.begin(plan, phase, per_core_M_max);
    if (!d.active()) {
        return 0;
    }
    uint32_t count[kMaxGroups] = {0};
    for (uint32_t e = 0; e < num_experts; ++e) {
        const uint32_t g = d.deal(tiles_of(e));
        if (g != kNoGroup) {
            ++count[g];
        }
    }
    uint32_t time[kMaxGroups] = {0};
    uint32_t seen[kMaxGroups] = {0};
    d.begin(plan, phase, per_core_M_max);
    for (uint32_t e = 0; e < num_experts; ++e) {
        const uint32_t tiles = tiles_of(e);
        const uint32_t g = d.deal(tiles);
        if (g == kNoGroup) {
            continue;
        }
        ++seen[g];
        uint32_t streams = 0;
        for (uint32_t h = 0; h < d.groups; ++h) {
            streams += (count[h] >= seen[g]) ? 1u : 0u;
        }
        time[g] += model_expert(tiles, d.rows, per_core_M_max, streams);
    }
    uint32_t m = 0;
    for (uint32_t g = 0; g < d.groups; ++g) {
        m = (time[g] > m) ? time[g] : m;
    }
    return m;
}

// The two-phase candidate: full waves on the best uniform size followed by the leftover experts on
// a larger uniform layout or a mixed-height one, when the model predicts a clear gain; otherwise
// rows_a == 0 (no phase A). A mixed phase B gives an odd leftover a group each without leaving rows
// idle: dsv3's 3 leftover 5-tile experts run per_core_M 2 on 4+3+3 rows (3 streams), where the
// uniform layouts either keep them at per_core_M 3 on 2-row groups or need two rounds on 5-row ones.
//
// Phase B must finish the leftover in ONE round (one leftover expert per phase-B group at most).
// A phase-B group's stream rate is capped by its 11 column senders, not by DRAM (r02-b03-a01:
// a lone 5-row group pulls ~250 GB/s), so a second phase-B round on a big group runs far slower
// than the model's DRAM share says, and loses to the uniform layout's one-round tail.
template <typename TilesOf>
inline Plan make_plan(
    const TilesOf& tiles_of,
    uint32_t num_experts,
    uint32_t per_core_M_max,
    const uint32_t (&uniform)[kNumUniformLayouts]) {
    Plan single;
    single.rows_b = pick_group_rows(uniform);
    const uint32_t rows_a = single.rows_b;
    if (rows_a >= kGridY) {
        return single;
    }
    const uint32_t groups_a = kGridY / rows_a;
    uint32_t active = 0;
    uint32_t t_max = 0;
    for (uint32_t e = 0; e < num_experts; ++e) {
        const uint32_t t = tiles_of(e);
        active += (t > 0) ? 1u : 0u;
        t_max = (t > t_max) ? t : t_max;
    }
    const uint32_t n_a = (active / groups_a) * groups_a;
    if (n_a == 0 || n_a == active) {
        return single;
    }
    // thresh = the n_a-th largest tile count; `above` experts have more.
    uint32_t thresh = t_max;
    uint32_t above = 0;
    while (true) {
        uint32_t at = 0;
        uint32_t next = 0;
        for (uint32_t e = 0; e < num_experts; ++e) {
            const uint32_t t = tiles_of(e);
            if (t == thresh) {
                ++at;
            } else if (t < thresh && t > next) {
                next = t;
            }
        }
        if (above + at >= n_a || next == 0) {
            break;
        }
        above += at;
        thresh = next;
    }
    // Ring alignment at the phase switch needs equal units per phase-A group: every phase-A expert
    // must take the same number of chunks (monotonic in tiles, so compare the extremes).
    const uint32_t chunk_a = per_core_M_max * rows_a;
    if (num_chunks(t_max, chunk_a) != num_chunks(thresh, chunk_a)) {
        return single;
    }
    Plan two;
    two.rows_a = rows_a;
    two.thresh = thresh;
    two.quota_eq = n_a - above;

    // The leftover experts (phase B), largest first. n_left < groups_a <= kMaxGroups.
    uint32_t left[kMaxGroups];
    uint32_t n_left = 0;
    {
        uint32_t eq_seen = 0;
        for (uint32_t e = 0; e < num_experts; ++e) {
            const uint32_t t = tiles_of(e);
            if (t == 0 || t > thresh || (t == thresh && eq_seen++ < two.quota_eq)) {
                continue;
            }
            uint32_t i = n_left++;
            for (; i > 0 && left[i - 1] < t; --i) {
                left[i] = left[i - 1];
            }
            left[i] = t;
        }
    }

    const uint32_t t_single = model_phase(tiles_of, num_experts, per_core_M_max, single, 1);
    const uint32_t t_a = model_phase(tiles_of, num_experts, per_core_M_max, two, 0);
    Plan best = single;
    uint32_t best_t = t_single * (100 - kModelMarginPct);
    // Phase-B candidates: every layout with a group per leftover expert (one round), other than phase
    // A's own. Uniform layouts come first, so a mixed one must be strictly better to replace them.
    const uint32_t layout_a = uniform_layout_index(rows_a);
    for (uint32_t li = 0; li < num_layouts(kGridY); ++li) {
        const GroupLayout& l = layout_at(kGridY, li);
        if (li == layout_a || l.groups < n_left || (li < kNumUniformLayouts && l.rows[0] < rows_a)) {
            continue;
        }
        uint32_t t_b = 0;
        for (uint32_t k = 0; k < n_left; ++k) {
            const uint32_t t = model_expert(left[k], l.rows[k], per_core_M_max, n_left);
            t_b = (t > t_b) ? t : t_b;
        }
        const uint32_t t_two = (t_a + t_b + kModelPhaseSwitch) * 100;
        if (t_two < best_t) {
            best_t = t_two;
            best = two;
            best.rows_b = l.rows[0];
            best.layout_b = li;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------------------------
// The launch plan every kernel runs: up to two phases, each a row-group layout (index into the
// layout table, which the host's per-layout topology sets follow) and owner[phase][e] = the group
// (within that phase's layout) that runs local expert e in that phase, kNoGroup otherwise.
//
// A two-phase plan (make_plan) is taken when the model says it clearly beats the uniform layouts.
// Otherwise phase A is empty and phase B is plan_groups' single layout, uniform or mixed-height.
constexpr uint32_t kNoLayout = 0xFFFFFFFFu;

struct LaunchPlan {
    uint32_t layout[2] = {kNoLayout, 0};
};

//
// The planner walks the tile counts a few hundred times (every layout candidate, every model pass).
// `tiles_of` reads the counts and idx pages in L1 and band-filters and clamps each entry, so the
// counts are read ONCE into a local array here and every pass walks that instead. Every RISC runs
// this before its first DRAM read, so it is critical-path time on every call.
template <uint32_t NumExperts, typename TilesOfL1>
inline LaunchPlan plan_launch(const TilesOfL1& tiles_of_l1, uint32_t per_core_M_max, uint8_t (&owner)[2][NumExperts]) {
    uint16_t tiles_cache[NumExperts];
    for (uint32_t e = 0; e < NumExperts; ++e) {
        tiles_cache[e] = static_cast<uint16_t>(tiles_of_l1(e));
    }
    auto tiles_of = [&](uint32_t e) -> uint32_t { return tiles_cache[e]; };
    uint32_t uniform[kNumUniformLayouts];
    uniform_makespans(tiles_of, NumExperts, per_core_M_max, uniform);
    LaunchPlan lp;
    const Plan two = make_plan(tiles_of, NumExperts, per_core_M_max, uniform);
    if (two.rows_a == 0) {
        for (uint32_t e = 0; e < NumExperts; ++e) {
            owner[0][e] = kNoGroup;
        }
        lp.layout[1] = plan_groups(tiles_of, per_core_M_max, uniform, owner[1]);
        return lp;
    }
    lp.layout[0] = uniform_layout_index(two.rows_a);
    lp.layout[1] = two.layout_b;
    PhaseDealer d;
    d.begin(two, 0, per_core_M_max);
    for (uint32_t e = 0; e < NumExperts; ++e) {
        owner[0][e] = static_cast<uint8_t>(d.deal(tiles_of(e)));
    }
    // Phase B: the leftover experts (fewer than phase A's groups) in expert order; the k-th largest
    // (ties -> lower index) runs on group k, matching make_plan's model.
    uint32_t left[kMaxGroups];
    uint32_t left_tiles[kMaxGroups];
    uint32_t n_left = 0;
    for (uint32_t e = 0; e < NumExperts; ++e) {
        owner[1][e] = kNoGroup;
        const uint32_t t = tiles_of(e);
        if (t != 0 && owner[0][e] == kNoGroup && n_left < kMaxGroups) {
            left[n_left] = e;
            left_tiles[n_left] = t;
            ++n_left;
        }
    }
    for (uint32_t i = 0; i < n_left; ++i) {
        uint32_t rank = 0;
        for (uint32_t j = 0; j < n_left; ++j) {
            rank += (left_tiles[j] > left_tiles[i] || (left_tiles[j] == left_tiles[i] && j < i)) ? 1u : 0u;
        }
        owner[1][left[i]] = static_cast<uint8_t>(rank);
    }
    return lp;
}

#if defined(COMPILE_FOR_TRISC) || defined(COMPILE_FOR_BRISC)
#pragma GCC pop_options
#endif

}  // namespace adaptive_chunk
