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
// CBs are sized on the host to the MAX chunk, passed in as `max_chunk`
// (= per_core_M_max * kGridY: the factory requests per_core_M 4, widens it to 8
// only where the avoided weight traffic pays for it, and its L1 guard may lower
// it). The picker never returns more than `max_chunk`, so the runtime
// per_core_M always fits the allocated CBs; a smaller pick simply uses fewer of
// the reserved tiles and shrinks every M-dimension loop.
//
// The three kernels MUST derive identical chunk_M_tiles / per_core_M / chunk
// counts from the same count, or the reader/compute/writer disagree on the row
// mapping and FFN output lands on the wrong token rows. This is the single
// source of truth. Pure integer arithmetic (no NoC/CB/reg APIs) so the header
// is valid in BRISC, NCRISC, and TRISC translation units alike.
namespace adaptive_chunk {

constexpr uint32_t kGridY = 8;  // M-row cores; a chunk spans per_core_M * kGridY tile-rows

// Chunk layout for `count_tiles` tile-rows, given the CB-sized maximum chunk
// `max_chunk` (= per_core_M_max * kGridY): a run of FULL chunks of max_chunk,
// then ONE tail chunk sized down to the remainder. This minimizes the chunk
// count (each chunk re-reads the full gate/up/down weights, the DRAM bottleneck)
// AND minimizes phantom M-work (the tail covers only the leftover rows), so
// per_core_M adapts to the actual token count per chunk with essentially no
// wasted rows — e.g. with max_chunk 64, 160 tiles -> 64 + 64 + 32
// (per_core_M 8,8,4), 3 chunks, zero phantom.
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

// ---------------------------------------------------------------------------------------
// WAVES. An expert whose tokens fit a few tile-rows leaves most of the 8 grid rows idle
// (rows past its count still MAC stale data in lockstep), and every expert pays the full
// per-K-block weight-stream and handshake cost. Small experts are instead run AT ONCE in a
// wave, each on its own slice of grid rows with its own weight stream:
//   HALF (pair)     2 experts, rows [0, 4) and [4, 8), a weight sender per column and half;
//   QUARTER (quad)  4 experts on 2-row groups, a sender per column and group;
//   EIGHTH          8 experts, one row each, every core reading its own weights;
//   PARTIAL EIGHTH  the last 4-7 small experts, one row each (see below).
// An expert on r rows does ceil(count / r) tile-rows of M-work per core instead of
// ceil(count / 8).
//
// The layout is chosen per expert from the DEVICE counts, identically in all three kernels:
// consecutive non-empty experts (local order) are grouped when all fit ONE chunk of their
// slice and the cost model below says the wave is no slower than the smaller layouts. Anything larger (the hot experts
// of skewed routing) keeps the whole grid.
//
// CB lockstep: a paired expert runs exactly one chunk on each half, so after the pair every
// core has advanced every CB by exactly one chunk -- the same as for a whole-grid chunk --
// and whole-grid multicasts that follow still land on matching ring slots.
constexpr uint32_t kHalfRows = kGridY / 2;
// QUARTER: four small experts at once, one per 2-row group [2q, 2q + 2), each group with its own
// per-column weight sender. Same lockstep rule as the halves: one chunk per grouped expert.
constexpr uint32_t kQuarterRows = kGridY / 4;
// EIGHTH: eight small experts at once, one per grid row. Every core is its own weight sender
// (no column multicast, no ready/valid handshake), so the wave carries 8 weight streams.
constexpr uint32_t kEighthRows = 1;
enum WaveRole : uint32_t {
    kRoleSkip = 0,
    kRoleFull = 1,
    kRoleTop = 2,       // half 0 (rows 0-3)
    kRoleBottom = 3,    // half 1 (rows 4-7)
    kRoleQuarter0 = 4,  // quarter q = role - kRoleQuarter0 (rows 2q, 2q + 1)
    kRoleEighth0 = 8,   // eighth r = role - kRoleEighth0 (row r)
};

inline uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

inline bool pair_on_halves(uint32_t a, uint32_t b, uint32_t per_core_M_max) {
    const uint32_t cap = kHalfRows * per_core_M_max;
    if (a == 0 || b == 0 || a > cap || b > cap) {
        return false;
    }
    const uint32_t ha = ceil_div(a, kHalfRows);
    const uint32_t hb = ceil_div(b, kHalfRows);
    const uint32_t fa = ceil_div(a, kGridY);
    const uint32_t fb = ceil_div(b, kGridY);
    return ((ha > hb) ? ha : hb) <= fa + fb;
}

// Relative cost (tenths of one per-core tile-row of the three matmuls) of what the pair rule
// would do with two experts: one half-grid pair, or two whole-grid experts one after the other.
// The fixed terms model the per-wave weight stream + handshakes, which grow with the number of
// concurrent streams (measured: 2 / 2.8 / ~5 tile-rows for 1 / 2 / 4 streams).
inline uint32_t pair_cost(uint32_t a, uint32_t b, uint32_t per_core_M_max) {
    if (pair_on_halves(a, b, per_core_M_max)) {
        const uint32_t ha = ceil_div(a, kHalfRows);
        const uint32_t hb = ceil_div(b, kHalfRows);
        return 28 + 10 * ((ha > hb) ? ha : hb);
    }
    return 40 + 10 * (ceil_div(a, kGridY) + ceil_div(b, kGridY));
}

// Cost of four experts as a quad on the 2-row groups, or ~0u when one does not fit a quarter chunk.
inline uint32_t quad_cost(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t per_core_M_max) {
    const uint32_t cap = kQuarterRows * per_core_M_max;
    if (a > cap || b > cap || c > cap || d > cap) {
        return ~0u;
    }
    uint32_t m = a;
    m = (b > m) ? b : m;
    m = (c > m) ? c : m;
    m = (d > m) ? d : m;
    return 50 + 10 * ceil_div(m, kQuarterRows);
}

// Four experts on the four 2-row groups when all fit one quarter chunk and the wave is no
// slower than the two pairs (or whole-grid runs) the pair rule would otherwise use.
inline bool quad_on_quarters(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t per_core_M_max) {
    const uint32_t quad = quad_cost(a, b, c, d, per_core_M_max);
    return quad != ~0u && quad <= pair_cost(a, b, per_core_M_max) + pair_cost(c, d, per_core_M_max);
}

// Cheapest of the quad / pair plans for four experts (what the planner would do without eighths).
inline uint32_t four_cost(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t per_core_M_max) {
    const uint32_t pairs = pair_cost(a, b, per_core_M_max) + pair_cost(c, d, per_core_M_max);
    const uint32_t quad = quad_cost(a, b, c, d, per_core_M_max);
    return (quad < pairs) ? quad : pairs;
}

// Eight experts on the eight grid rows when all fit one row chunk and the wave is no slower than
// the quarter/pair plan. Fixed term extrapolates the stream cost (2 / 2.8 / 5 tile-rows for
// 1 / 2 / 4 streams) to 8 streams with no column multicast or handshakes.
inline bool eight_on_rows(const uint32_t* c, uint32_t per_core_M_max) {
    uint32_t m = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        if (c[i] == 0 || c[i] > per_core_M_max) {
            return false;
        }
        m = (c[i] > m) ? c[i] : m;
    }
    const uint32_t eight = 92 + 10 * m;
    return eight <=
           four_cost(c[0], c[1], c[2], c[3], per_core_M_max) + four_cost(c[4], c[5], c[6], c[7], per_core_M_max);
}

// PARTIAL EIGHTH: the LAST n (4..7) non-empty experts of the program, each on its own grid row
// (spread over the 8 rows), every core streaming its own weights as in a full eighth wave; the
// other rows sit idle. Without it this tail runs as a quad (+ pair) with column-multicast weight
// feeds, whose 2-row groups finish up to 40% apart at random (measured per-core end times on
// 4-expert, 1K-token cases), while eighth waves finish within ~2% row to row.
// Only where the experts are small enough (<= kPartialEighthMaxM tile-rows) that one row's matmul
// stays under the wave's weight stream, so giving up the second row costs nothing, AND where the
// host set PARTIAL_EIGHTH: one core's weight slice per expert is small enough to stream from one
// core (measured: +15% at 1.3 MB of bf4 weights per core and expert, -5% at 3 MB, -3% at
// 3.5 MB).
// LAST WAVE ONLY: idle rows do not advance their CB rings, so a later multi-row (column
// multicast) layout would see ring slots that differ between rows. Nothing follows the last wave.
constexpr uint32_t kPartialEighthMinExperts = 4;
constexpr uint32_t kPartialEighthMaxM = 3;
inline uint32_t partial_eighth_row(uint32_t i, uint32_t n) { return (i * kGridY) / n; }
inline bool partial_eighth_on_rows(const uint32_t* c, uint32_t n, uint32_t per_core_M_max) {
    if (n < kPartialEighthMinExperts || n >= kGridY) {
        return false;
    }
    const uint32_t cap = (per_core_M_max < kPartialEighthMaxM) ? per_core_M_max : kPartialEighthMaxM;
    for (uint32_t i = 0; i < n; ++i) {
        if (c[i] == 0 || c[i] > cap) {
            return false;
        }
    }
    return true;
}

// Streaming planner: call role() once per local expert, in increasing order. tiles_of(e)
// must return the expert's CLAMPED tile-row count (0 if empty or outside the band).
// A group (quad or pair) is decided when its first expert comes up; its later members are
// queued in pend_* and handed out as they come up (empties in between stay kRoleSkip).
struct WavePlanner {
    uint32_t pend_e[7] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    uint32_t pend_role[7] = {0, 0, 0, 0, 0, 0, 0};
    uint32_t n_pend = 0;
    uint32_t i_pend = 0;

    template <class TilesOf>
    uint32_t role(uint32_t e, uint32_t num_experts, uint32_t per_core_M_max, const TilesOf& tiles_of) {
        const uint32_t c = tiles_of(e);
        if (c == 0) {
            return kRoleSkip;
        }
        if (i_pend < n_pend && pend_e[i_pend] == e) {
            return pend_role[i_pend++];
        }
        n_pend = 0;
        i_pend = 0;
        // Next (up to) seven non-empty experts after e.
        uint32_t nx[7] = {0, 0, 0, 0, 0, 0, 0};
        uint32_t nc[8] = {c, 0, 0, 0, 0, 0, 0, 0};
        uint32_t found = 0;
        for (uint32_t n = e + 1; n < num_experts && found < 7; ++n) {
            const uint32_t cn = tiles_of(n);
            if (cn != 0) {
                nx[found] = n;
                nc[found + 1] = cn;
                ++found;
            }
        }
        if (found == 7 && eight_on_rows(nc, per_core_M_max)) {
            for (uint32_t i = 0; i < 7; ++i) {
                pend_e[i] = nx[i];
                pend_role[i] = kRoleEighth0 + 1 + i;
            }
            n_pend = 7;
            return kRoleEighth0;
        }
        // found < 7: the scan reached the last expert, so these are ALL the remaining ones.
#if defined(PARTIAL_EIGHTH)
        if (found < 7 && partial_eighth_on_rows(nc, found + 1, per_core_M_max)) {
            for (uint32_t i = 0; i < found; ++i) {
                pend_e[i] = nx[i];
                pend_role[i] = kRoleEighth0 + partial_eighth_row(i + 1, found + 1);
            }
            n_pend = found;
            return kRoleEighth0 + partial_eighth_row(0, found + 1);
        }
#endif
        if (found >= 3 && quad_on_quarters(c, nc[1], nc[2], nc[3], per_core_M_max)) {
            for (uint32_t i = 0; i < 3; ++i) {
                pend_e[i] = nx[i];
                pend_role[i] = kRoleQuarter0 + 1 + i;
            }
            n_pend = 3;
            return kRoleQuarter0;
        }
        if (found >= 1 && pair_on_halves(c, nc[1], per_core_M_max)) {
            pend_e[0] = nx[0];
            pend_role[0] = kRoleBottom;
            n_pend = 1;
            return kRoleTop;
        }
        return kRoleFull;
    }
};

// Does a core in grid row `gy` run an expert with this role, and on how many rows.
inline uint32_t role_rows(uint32_t role) {
    return (role == kRoleFull)       ? kGridY
           : (role >= kRoleEighth0)  ? kEighthRows
           : (role >= kRoleQuarter0) ? kQuarterRows
                                     : kHalfRows;
}
inline bool role_runs_here(uint32_t role, uint32_t gy) {
    if (role == kRoleSkip) {
        return false;
    }
    if (role == kRoleFull) {
        return true;
    }
    if (role >= kRoleEighth0) {
        return gy == role - kRoleEighth0;
    }
    if (role >= kRoleQuarter0) {
        return gy / kQuarterRows == role - kRoleQuarter0;
    }
    return gy / kHalfRows == role - kRoleTop;
}
inline uint32_t role_row(uint32_t role, uint32_t gy) { return gy % role_rows(role); }

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
// `rows` is the number of grid rows the expert runs on: kGridY on the whole grid, fewer in a
// wave (see WavePlanner).
inline uint32_t per_core_M_for_chunk(uint32_t c, uint32_t count_tiles, uint32_t max_chunk, uint32_t rows) {
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

}  // namespace adaptive_chunk
