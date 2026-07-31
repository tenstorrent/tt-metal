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

// Tile-rows that core `my_col` injects into the x row-multicast for a block of `m_eff` tile-rows.
// Round `t`'s rotating injector is column `t % hgroups`, so this counts t in [0, m_eff) with
// `t % hgroups == my_col`. Shared so the reader's staging loop and compute's fused-tilize count
// cannot disagree (it used to be a host-computed runtime arg fixed at M_BLOCK).
inline uint32_t inject_rows(uint32_t m_eff, uint32_t my_col, uint32_t hgroups) {
    return (my_col < m_eff) ? ((m_eff - my_col + hgroups - 1) / hgroups) : 0;
}

}  // namespace moe_fused_swiglu
