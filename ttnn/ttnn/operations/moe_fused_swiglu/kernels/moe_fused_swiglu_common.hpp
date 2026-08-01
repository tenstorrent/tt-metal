// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — declarations shared by ALL THREE kernels.
//
// Deliberately include-free so the COMPUTE translation unit can pull it in as well: a compute
// kernel must not see the dataflow API. The dataflow-only bank-run reader/writer lives next door in
// `moe_fused_swiglu_bank_runs.hpp`.
//
// SINGLE SOURCE OF TRUTH for the token-count mailbox word layout. The reader publishes it, and
// compute AND the writer read it, so the word indices must be written down exactly once (they were
// bare literals in three files before).

#pragma once

#include <cstdint>

namespace moe_fused_swiglu {

// L1 mailbox word layout. The reader fills 0..2 and then stamps MAGIC into word 3; every other
// kernel spins on word 3 and only then reads 0..2. One page (64 B) per core, zeroed host-side so a
// stale magic from a previous dispatch can never be mistaken for a fresh publish.
constexpr uint32_t MBOX_COUNT = 0;     // counts[idx[local_expert_id]] — the RUNTIME token count
constexpr uint32_t MBOX_M_T = 1;       // ceil(count/32), clamped to M_T_MAX
constexpr uint32_t MBOX_M_BLOCKS = 2;  // ceil(M_t / M_BLOCK) — the outer-loop trip count
constexpr uint32_t MBOX_READY = 3;     // == MAILBOX_MAGIC once words 0..2 are valid

// ---------------------------------------------------------------------------
// `m_tiles` — the RUNTIME token tile-rows worked per M-block (op_design.md §3).
//
// SINGLE SOURCE OF TRUTH, called from ALL THREE kernels: the reader's x-multicast round count and
// cb_x_tiles/cb_h increments, compute's MatmulBlockShape + loop bounds, and the writer's CB waits
// are THE SAME NUMBER. A disagreement of one tile deadlocks the collectives, so this is a pure
// function of (m_t, b, m_block, m_min) — all four identical on every core and every RISC-V — and
// the three kernels therefore agree bit-for-bit.
//
// The tail block is rounded UP to a power of two <= m_block, never down to an arbitrary `M_t`, for
// one hard reason: a CB reserve must never straddle its FIFO end, and every M-scaled CB is sized
// `DEPTH * m_block * W`. With m_block a power of two (host-asserted) a power-of-two m_eff divides
// it, so `m_eff * W` divides every such total and the FIFO write pointer stays block-aligned for
// any push order. `m_min` additionally keeps m_eff a multiple of the matmul's out_subblock_h.
//
// Rows [count, m_eff*32) are UNDEFINED tile padding by contract, so over-computing them is legal;
// what this saves is the work on the tile-rows past m_eff that the op used to do anyway
// (m_block = 8 tile-rows regardless of the count).
inline uint32_t m_tiles_eff(uint32_t m_t, uint32_t b, uint32_t m_block, uint32_t m_min) {
    const uint32_t done = b * m_block;
    const uint32_t rem = (m_t > done) ? (m_t - done) : 0;
    if (rem >= m_block) {
        return m_block;  // only the LAST block can shrink; every earlier one is full
    }
    uint32_t p = (m_min > 0) ? m_min : 1;
    while (p < rem) {
        p <<= 1;
    }
    return (p > m_block) ? m_block : p;
}

// ---------------------------------------------------------------------------
// PERF 2 — the REDUCE-SCATTER slice plan (`MOE_SWIGLU_REDUCE=scatter`).
//
// SINGLE SOURCE OF TRUTH, called from ALL THREE kernels with the SAME (t, cap), for the same reason
// m_tiles_eff() is: the column's all-to-all is only deadlock-free while every core agrees, to the
// tile, on who owns which slice. `t = m_eff * HN_PAD` is the runtime tile count of ONE gate/up block
// and `cap = KGROUPS` is the column height, so the plan is a pure function of the mailbox words and
// the grid — identical on every core and every RISC-V, and it SHRINKS with the runtime m_eff exactly
// as everything else in this op does.
//
// FLAT and UNIFORM by construction: the worker count is the LARGEST DIVISOR of `t` that is <= cap, so
// every worker owns exactly `t / w` tiles and cores [w, cap) are IDLE for the reduce (they still
// contribute their own partial). The ragged `ceil`/`floor` split measured 1-10% FASTER in the
// bake-off (it puts every core to work and shortens the critical slice) and is deliberately NOT
// shipped: unequal slice sizes force every slice CB to `lcm(a_min, a_max)` pages, and a slice CB
// whose page count is not a multiple of the per-pass push size walks its write pointer past the CB
// end and SILENTLY OVERRUNS INTO THE NEXT CB — measured PCC 0.709-0.886 for the ragged 5/5/../4/4
// plan against >= 0.9955 for every uniform one. See MOE_SWIGLU_REDUCE in the program descriptor.
inline uint32_t slice_workers(uint32_t t, uint32_t cap) {
    uint32_t w = (t < cap) ? t : cap;
    while (w > 1 && (t % w) != 0) {
        --w;
    }
    return w;
}

// Tiles the core at column row `row` owns of a `t`-tile block, 0 if it is an idle core. The slice is
// a CONTIGUOUS tile range at `row * (t / w)` because the gate/up block layout is `m * HN_PAD + n`
// (OUT_SUBBLOCK_H_GU == 1, SubblockMajor), which is what makes every gather leg ONE coalesced
// transaction instead of m_eff strided ones.
inline uint32_t slice_assigned(uint32_t t, uint32_t cap, uint32_t row) {
    const uint32_t w = slice_workers(t, cap);
    return (row < w) ? (t / w) : 0;
}

// Tile-rows that core `my_col` injects into the x row-multicast for a block of `m_eff` tile-rows.
// Round `t`'s rotating injector is column `t % hgroups`, so this counts t in [0, m_eff) with
// `t % hgroups == my_col`. Shared so the reader's staging loop and compute's fused-tilize count
// cannot disagree (it used to be a host-computed runtime arg fixed at M_BLOCK).
inline uint32_t inject_rows(uint32_t m_eff, uint32_t first, uint32_t hgroups) {
    return (first < m_eff) ? ((m_eff - first + hgroups - 1) / hgroups) : 0;
}

// PERF 13 — the ONE definition of a core's FIRST x-injection tile-row, shared by the reader's
// staging loop, the reader's multicast lane test and compute's fused-tilize count. Three sites
// derive the injector map; the reference op records the failure mode when they disagree ("skew must
// be 0 wherever a SECOND kernel independently recomputes this map and is not handed the same
// skew"), so the skew lives here and nowhere else.
//
// diag == 0: injector for tile-row t is column `t % hgroups` in EVERY row -> a VERTICAL LINE of
//            readers, measured upstream as the worst NOC0 read shape (789k vs 204k diagonal, 3.9x;
//            NOC0 routes east->south so a response turns south in the reader's own column).
// diag == 1: lane = (t + y) % hgroups -> core ((t + y) % hgroups, y), one injector per row on a
//            DIAGONAL. Equivalently this core's rows start at (my_col - y) mod hgroups.
inline uint32_t inject_first(uint32_t my_col, uint32_t my_row, uint32_t hgroups, uint32_t diag) {
    return diag ? ((my_col + hgroups - (my_row % hgroups)) % hgroups) : my_col;
}

}  // namespace moe_fused_swiglu
