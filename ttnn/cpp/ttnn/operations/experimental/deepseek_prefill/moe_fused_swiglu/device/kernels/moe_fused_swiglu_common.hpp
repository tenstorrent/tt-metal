// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

// Hidden-block geometry. Most grids use the original uniform-start layout: block r begins at
// r*HN_PAD and only the final block is ragged. If that would leave a worker column empty (for
// example HID_T=64, HGROUPS=12, HN_PAD=6), the host selects a balanced split. The predicate is
// derivable from the same compile-time constants in all three kernels, so no extra CT arg or
// runtime table is needed. Fixed HN_PAD-sized CB slots remain uniform; only their real prefix is
// read and multiplied.
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

// L1 mailbox word layout. The reader fills 0..2 and then stamps MAGIC into word 3; every other
// kernel spins on word 3 and only then reads 0..2. One page (64 B) per core, zeroed host-side so a
// stale magic from a previous dispatch can never be mistaken for a fresh publish.
constexpr uint32_t MBOX_COUNT = 0;     // counts[idx[local_expert_id]] — the RUNTIME token count
constexpr uint32_t MBOX_M_T = 1;       // ceil(count/32), clamped to M_T_MAX
constexpr uint32_t MBOX_M_BLOCKS = 2;  // ceil(M_t / M_BLOCK) — the outer-loop trip count
constexpr uint32_t MBOX_READY = 3;     // == MAILBOX_MAGIC once words 0..2, 4 are valid
// start[global_expert_id] in TOKEN rows — this expert's region base in a SHARED x/output buffer,
// 0 unless the caller passed expert_region_offsets. Published here rather than read twice because
// the WRITER needs it too, and it already spins on this mailbox: the reference C++ op has no such
// channel and pays a second DRAM page read plus a second accessor and scratch CB for it. Raw token
// rows, not tile rows — the row-major x read offsets STICKS while the tiled x read and the output
// write offset TILE rows, and each site divides by TILE_H itself.
constexpr uint32_t MBOX_START_ROW = 4;
// Writer-owned whole h rounds publish completion here after their linked NoC1 payload+flag chain
// has flushed.  One diagonal writer owns at most one round, so b+1 is a monotone per-core counter.
constexpr uint32_t MBOX_HSEND_DONE = 5;
// Reader -> writer, same core: the NoC0 up-scatter writes for block b have landed.  The optional
// single-signal scatter path lets the writer emit one completion only after this and its own gate
// payload are both complete, halving the destination atomic fan-in without merging CB ownership.
constexpr uint32_t MBOX_UP_SCATTER_DONE = 6;

// ---------------------------------------------------------------------------
// The mailbox handshake. The reader fills words 0..2 and then stamps MAGIC into word 3; the writer
// and all three compute TRISCs spin on word 3 and only then read 0..2. Written here once because
// the spin appeared in three files and differed only in how it invalidated the cache line.
//
// Raw L1 rather than a CB because the M-block trip count must be identical on all three TRISCs and
// `cb_wait_front` in a compute kernel is UNPACK-only, so a CB handoff would let MATH and PACK
// diverge from UNPACK.
struct Mailbox {
    uint32_t count, m_t, m_blocks, start_row;
};

//: The reader's publish. The fence orders every payload word BEFORE the magic, so a peer that
//: sees the magic sees the payload.
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

//: The consumer side. `invalidate` is the dataflow kernels' cache invalidation; compute passes a
//: plain fence instead, because it must not see the dataflow API.
template <class Invalidate>
inline Mailbox mailbox_wait(uint32_t addr, uint32_t magic, Invalidate invalidate) {
    volatile tt_l1_ptr uint32_t* mb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    while (mb[MBOX_READY] != magic) {
        invalidate();
    }
    return Mailbox{mb[MBOX_COUNT], mb[MBOX_M_T], mb[MBOX_M_BLOCKS], mb[MBOX_START_ROW]};
}

// ---------------------------------------------------------------------------
// `m_tiles` — the RUNTIME token tile-rows worked per M-block (op_design.md §3).
//
// SINGLE SOURCE OF TRUTH, called from ALL THREE kernels: the reader's x-multicast round count and
// cb_x_tiles/cb_h increments, compute's matmul shape + loop bounds, and the writer's CB waits
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

// `m_tiles_real` — the tile-rows of this block that carry REAL tokens: m_tiles_eff() WITHOUT the
// power-of-two round-up. Guarantees 1 <= m_tiles_real <= m_tiles_eff for every block the M-loop
// actually runs.
//
// This is the ARITHMETIC count, never a page count. m_tiles_eff() stays the unit of every CB
// reserve/push/pop and of the reduce-scatter slice plan, because those are what must divide M_BLOCK
// and agree across cores without communication. The FPU work does not have to: the gate/up output
// block is m-MAJOR, so the real rows are a contiguous PREFIX, and a matmul over `m_real` sub-blocks
// of height OUT_SUBBLOCK_H_GU writes exactly that prefix and leaves the pad rows stale. Rows
// [count, m_eff*32) are UNDEFINED tile padding by contract — the op used to fill them with
// silu(pad @ Wg) * (pad @ Wu) @ Wd, which is undefined in exactly the same way.
//
// WHICH CONSUMERS THIS IS SAFE FOR is not a free choice; see the m_rows comment in the compute
// kernel. gate/up owns its own in0 lifecycle, so shrinking its shape shrinks only a wait_front.
// `down` does not, and shrinking ITS shape shrank a cb_pop_front and hung the op.
//
// `rem` cannot be 0 for a block the loop runs: m_blocks = ceil(m_t / m_block), so b < m_blocks
// implies b * m_block < m_t. The clamp is still here rather than an ASSERT because a 0-row matmul
// shape is not a crash — it is a silently skipped block — and this is a three-kernel contract.
inline uint32_t m_tiles_real(uint32_t m_t, uint32_t b, uint32_t m_block) {
    const uint32_t done = b * m_block;
    const uint32_t rem = (m_t > done) ? (m_t - done) : 0;
    if (rem == 0) {
        return 1;
    }
    return (rem > m_block) ? m_block : rem;
}

// Smallest multiple of `mult` that is >= v, capped at `cap`. Keeps a shrunk row count legal for a
// sub-block height > 1: the matmul does `rows / height` sub-blocks and that division must be exact,
// so the shrink rounds UP to the height rather than truncating rows away. At the shipped
// OUT_SUBBLOCK_H_GU == 1 it is the identity; it is what makes the shrink safe if that is ever raised.
inline uint32_t round_up_capped(uint32_t v, uint32_t mult, uint32_t cap) {
    const uint32_t r = ((v + mult - 1) / mult) * mult;
    return (r > cap) ? cap : r;
}

// ---------------------------------------------------------------------------
// Reduce-scatter slice plan.
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

// The ONE definition of a core's FIRST x-injection tile-row, shared by the reader's staging loop,
// the reader's multicast lane test and compute's fused-tilize count. Three sites derive the
// injector map and a disagreement deadlocks the row multicast, so it lives here and nowhere else.
// Round t's injector is column `t % hgroups`, so this core's rows start at `my_col`.
inline uint32_t inject_first(uint32_t my_col) { return my_col; }

}  // namespace moe_fused_swiglu
