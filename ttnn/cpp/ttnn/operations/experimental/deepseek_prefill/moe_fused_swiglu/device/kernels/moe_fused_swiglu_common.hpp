// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — declarations shared by ALL THREE kernels.
//
// Deliberately include-free so the COMPUTE translation unit can pull it in too: a compute kernel
// must not see the dataflow API. The dataflow-only bank-run reader/writer is in
// `moe_fused_swiglu_bank_runs.hpp`.
//
// EVERYTHING HERE IS A THREE-KERNEL CONTRACT. Each function is a pure function of the mailbox words
// and compile-time constants, so reader, compute and writer derive identical values without
// communicating. A disagreement of one tile deadlocks the collectives, which is why these live in
// one file rather than being re-derived per kernel.
//

#pragma once

#include <cstdint>

namespace moe_fused_swiglu {

// Hidden-block geometry

// Most grids use uniform starts (block r at r*hn_pad, only the last ragged). If that would leave a
// worker column empty (e.g. hid_t=64, hgroups=12, hn_pad=6) the host picks a balanced split. The
// predicate is derivable from the same constants in all three kernels, so it needs no CT arg.
constexpr bool hidden_blocks_are_balanced(uint32_t hid_t, uint32_t hgroups, uint32_t hn_pad) {
    return hn_pad * (hgroups - 1) >= hid_t;
}

constexpr uint32_t hidden_block_start(uint32_t block, uint32_t hid_t, uint32_t hgroups, uint32_t hn_pad) {
    if (!hidden_blocks_are_balanced(hid_t, hgroups, hn_pad)) {
        return block * hn_pad;
    }
    const uint32_t base = hid_t / hgroups;
    const uint32_t rem = hid_t % hgroups;
    return block * base + ((block < rem) ? block : rem);
}

constexpr uint32_t hidden_block_rows(uint32_t block, uint32_t hid_t, uint32_t hgroups, uint32_t hn_pad) {
    if (!hidden_blocks_are_balanced(hid_t, hgroups, hn_pad)) {
        const uint32_t start = block * hn_pad;
        return (start + hn_pad > hid_t) ? (hid_t - start) : hn_pad;
    }
    return hid_t / hgroups + ((block < (hid_t % hgroups)) ? 1 : 0);
}

// The token-count mailbox
//
// Raw L1 rather than a CB because the M-block trip count must reach ALL THREE compute TRISCs, and
// `cb_wait_front` in a compute kernel is UNPACK-only — a CB handoff would let MATH and PACK diverge
// from UNPACK. One 64 B page per core, host-zeroed so a stale magic cannot be read as fresh.
//
// Protocol: the reader fills words 0..2 and 4..6, fences, then stamps MAGIC into word 3. Every peer
// spins on word 3 and only then reads the rest.

constexpr uint32_t MBOX_COUNT = 0;     // counts[idx[local_expert_id]] — the RUNTIME token count
constexpr uint32_t MBOX_M_T = 1;       // ceil(count/32), clamped to M_T_MAX
constexpr uint32_t MBOX_M_BLOCKS = 2;  // ceil(M_t / M_BLOCK) — the outer-loop trip count
constexpr uint32_t MBOX_READY = 3;     // == MAILBOX_MAGIC once the others are valid

// start[global_expert_id] in TOKEN rows: this expert's base in a SHARED x/output buffer, 0 unless
// the caller passed expert_region_offsets. Published here because the WRITER needs it too and
// already spins on this mailbox. Raw token rows — each site divides by TILE_H itself.
constexpr uint32_t MBOX_START_ROW = 4;

// Writer-owned whole-h rounds publish completion here once their linked NoC1 chain has flushed.
// One diagonal writer owns at most one round, so block_idx+1 is a monotone per-core counter.
constexpr uint32_t MBOX_HSEND_DONE = 5;

// Reader -> writer, same core: this block's NoC0 up-scatter writes have landed. Lets the writer emit
// one completion for both halves, halving the destination atomic fan-in.
constexpr uint32_t MBOX_UP_SCATTER_DONE = 6;

struct Mailbox {
    uint32_t count, m_t, m_blocks, start_row;
};

//: The fence orders every payload word BEFORE the magic, so a peer that sees the magic sees the
//: payload.
inline void mailbox_publish(
    uint32_t addr, uint32_t magic, uint32_t count, uint32_t m_t, uint32_t m_blocks, uint32_t start_row) {
    volatile tt_l1_ptr uint32_t* mb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    mb[MBOX_COUNT] = count;
    mb[MBOX_M_T] = m_t;
    mb[MBOX_M_BLOCKS] = m_blocks;
    mb[MBOX_START_ROW] = start_row;
    asm volatile("fence" ::: "memory");
    mb[MBOX_READY] = magic;
}

//: `invalidate` is the dataflow kernels' cache invalidation; compute passes a plain fence, because
//: it must not see the dataflow API.
template <class Invalidate>
inline Mailbox mailbox_wait(uint32_t addr, uint32_t magic, Invalidate invalidate) {
    volatile tt_l1_ptr uint32_t* mb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    while (mb[MBOX_READY] != magic) {
        invalidate();
    }
    return Mailbox{mb[MBOX_COUNT], mb[MBOX_M_T], mb[MBOX_M_BLOCKS], mb[MBOX_START_ROW]};
}

// Runtime M-block sizing

// PAGE count for this block: the tile-rows every CB reserve, push and pop is denominated in, and
// unit of the reduce-scatter slice plan.
//
// The tail rounds UP to a power of two so it still divides the `DEPTH * m_block * W` CB sizes and
// the write pointer stays block-aligned; `m_min` keeps it a multiple of out_subblock_h. Rows past
// `count` are UNDEFINED by contract, so over-computing them is legal.
inline uint32_t m_tiles_eff(uint32_t m_t, uint32_t block_idx, uint32_t m_block, uint32_t m_min) {
    const uint32_t done = block_idx * m_block;
    const uint32_t rem = (m_t > done) ? (m_t - done) : 0;
    if (rem >= m_block) {
        return m_block;  // only the LAST block can shrink
    }
    uint32_t p = (m_min > 0) ? m_min : 1;
    while (p < rem) {
        p <<= 1;
    }
    return (p > m_block) ? m_block : p;
}

// ARITHMETIC count: the tile-rows of this block carrying REAL tokens, i.e. m_tiles_eff() without the
// power-of-two round-up. Never a page count.
//
// SAFE FOR GATE/UP ONLY: its block is m-MAJOR (the real rows are a contiguous prefix) and it owns
// its in0 lifecycle, so a smaller shape shrinks only a wait_front. `down` derives its POP from the
// same shape, so a smaller shape under-pops, drifts the CB, and deadlocks.
inline uint32_t m_tiles_real(uint32_t m_t, uint32_t block_idx, uint32_t m_block) {
    const uint32_t done = block_idx * m_block;
    const uint32_t rem = (m_t > done) ? (m_t - done) : 0;
    if (rem == 0) {
        return 1;  // unreachable for a block the loop runs; a 0-row shape would silently skip work
    }
    return (rem > m_block) ? m_block : rem;
}

// Smallest multiple of `mult` >= v, capped. Keeps a shrunk row count legal for a sub-block height
// > 1, where `rows / height` must divide exactly. Identity at the shipped OUT_SUBBLOCK_H_GU == 1.
inline uint32_t round_up_capped(uint32_t v, uint32_t mult, uint32_t cap) {
    const uint32_t r = ((v + mult - 1) / mult) * mult;
    return (r > cap) ? cap : r;
}

// The reduce-scatter slice plan
//
// `t = m_eff * HN_PAD` tiles of one gate/up block over `cap = KGROUPS` column rows. The column's
// all-to-all is deadlock-free only while every core agrees to the tile on who owns which slice, so
// this is a pure function of the mailbox words and the grid.

// FLAT AND UNIFORM by construction: the worker count is the LARGEST DIVISOR of `t` that is <= cap,
// so every worker owns exactly `t / w` tiles and rows [w, cap) idle for the reduce.
//
// A ragged ceil/floor split would put every core to work, but its unequal slices leave a CB page
// count that is not a multiple of the per-pass push. The write pointer then walks past the CB end
// and overruns the next CB — silent corruption, not a hang.
inline uint32_t slice_workers(uint32_t t, uint32_t cap) {
    uint32_t w = (t < cap) ? t : cap;
    while (w > 1 && (t % w) != 0) {
        --w;
    }
    return w;
}

// Tiles owned by column row `row`, 0 if it is an idle core. The slice is a CONTIGUOUS tile range,
// because the gate/up block layout is `m * HN_PAD + n` — which is what makes every gather leg ONE
// coalesced transaction instead of m_eff strided ones.
inline uint32_t slice_assigned(uint32_t t, uint32_t cap, uint32_t row) {
    const uint32_t w = slice_workers(t, cap);
    return (row < w) ? (t / w) : 0;
}

// The x row-multicast injector map

// Round `t`'s injector is column `t % hgroups`, so this core's rows start at its own column.
// Three sites derive this map and a disagreement deadlocks the multicast.
inline uint32_t inject_first(uint32_t my_col) { return my_col; }

// Tile-rows core `my_col` injects for a block of `m_eff` rows: the count of t in [0, m_eff) with
// `t % hgroups == my_col`. Shared so the reader's staging loop and compute's fused-tilize count
// cannot disagree.
inline uint32_t inject_rows(uint32_t m_eff, uint32_t first, uint32_t hgroups) {
    return (first < m_eff) ? ((m_eff - first + hgroups - 1) / hgroups) : 0;
}

}  // namespace moe_fused_swiglu
