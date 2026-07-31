// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — WRITER (NoC1).
//
// Owns, per M-block:
//   1. the W_up weight stream — the NoC1 twin of the reader's W_gate stream (op_design.md §1.5
//      dual-issue split: a phase with two independent weight streams uses BOTH data-movement
//      RISC-Vs / both NoCs);
//   2. the CHILD side of the gate/up cross-column reduce tree (wait for the parent's invite,
//      unicast both partials into the parent's cb_reduce_*_in, signal);
//   3. the coalesced bank-run output write-back, clamped to tile-rows < ceil_tile(count) so rows
//      past the real token count are never touched.
//
// Raw-dataflow deviations are the same two documented at the head of the reader: bank-run
// noc_async_write/read for coalescing (no in-tree helper expresses a multi-page contiguous
// transaction on an interleaved tensor), and raw unicast + counting semaphores for the tree edge
// (mcast_pipe's SenderPipe is a rectangle multicast, not a point-to-point tree edge).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "moe_fused_swiglu_bank_runs.hpp"  // the ONE definition of the bank-run coalescing
#include "moe_fused_swiglu_common.hpp"     // the ONE definition of the mailbox word layout

constexpr uint32_t EMB_T = get_compile_time_arg_val(0);
constexpr uint32_t HID_T = get_compile_time_arg_val(1);
constexpr uint32_t KR_PAD = get_compile_time_arg_val(2);
constexpr uint32_t HN_PAD = get_compile_time_arg_val(3);
constexpr uint32_t EC_MAX = get_compile_time_arg_val(4);  // phase-2 N stride (uniform CB increment)
constexpr uint32_t M_BLOCK = get_compile_time_arg_val(5);
constexpr uint32_t HGROUPS = get_compile_time_arg_val(6);
constexpr uint32_t KGROUPS = get_compile_time_arg_val(7);
constexpr uint32_t NUM_BANKS = get_compile_time_arg_val(8);
constexpr uint32_t WRUN = get_compile_time_arg_val(9);
constexpr uint32_t SEM_GO = get_compile_time_arg_val(10);
constexpr uint32_t SEM_DATA = get_compile_time_arg_val(11);
constexpr uint32_t BFP4_TILE = get_compile_time_arg_val(12);
constexpr uint32_t BFP8_TILE = get_compile_time_arg_val(13);
constexpr uint32_t REMAP = get_compile_time_arg_val(14);
constexpr uint32_t MAILBOX_MAGIC = get_compile_time_arg_val(15);
// Smallest legal `m_eff` (= the matmul's out_subblock_h rounded up to a power of two). One
// host-side definition, passed identically to all three kernels — see m_tiles_eff().
constexpr uint32_t M_EFF_MIN = get_compile_time_arg_val(16);
// Concurrent child landing slots in the parent's cb_reduce_*_in (Refinement 2 lever 1) — the child
// needs it only to stride its own slot index into the parent's CB.
constexpr uint32_t REDUCE_SLOTS = get_compile_time_arg_val(17);
// REFINEMENT 3 — cross-M-block weight residency, the NoC1 half. W_up's read below carries no
// M-block index, so every block after the first re-reads bytes still resident in `cb_w_up`'s single
// slot. See the reader's W_RESIDENT comment for why the reserve/push handshake stays untouched.
constexpr uint32_t W_RESIDENT = get_compile_time_arg_val(18);

constexpr uint32_t cb_w_up = get_compile_time_arg_val(19);
constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(20);
constexpr uint32_t cb_gate_send = get_compile_time_arg_val(21);
constexpr uint32_t cb_up_send = get_compile_time_arg_val(22);
constexpr uint32_t cb_reduce_gate_in = get_compile_time_arg_val(23);
constexpr uint32_t cb_reduce_up_in = get_compile_time_arg_val(24);

constexpr uint32_t TA_BASE = 25;
constexpr auto wu_args = TensorAccessorArgs<TA_BASE>();
constexpr auto out_args = TensorAccessorArgs<wu_args.next_compile_time_args_offset()>();

// Bank-run coalescing (see moe_fused_swiglu_bank_runs.hpp): ONE definition, bound here to this
// kernel's compile-time knobs — identical to the reader's binding, which is the point.
using BR = moe_fused_swiglu::BankRuns<REMAP != 0, NUM_BANKS, WRUN>;

#ifdef MOE_SWIGLU_ZONES
#define MOE_ZONE(n) DeviceZoneScopedN(n)
#else
#define MOE_ZONE(n) ((void)0)
#endif

void kernel_main() {
    const uint32_t mailbox_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_up_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t kr = get_arg_val<uint32_t>(3);
    const uint32_t kstart = get_arg_val<uint32_t>(4);
    const uint32_t hstart = get_arg_val<uint32_t>(5);
    const uint32_t hn = get_arg_val<uint32_t>(6);
    const uint32_t ec = get_arg_val<uint32_t>(7);
    const uint32_t jstart = get_arg_val<uint32_t>(8);
    const uint32_t is_root = get_arg_val<uint32_t>(9);
    const uint32_t parent_x = get_arg_val<uint32_t>(10);
    const uint32_t parent_y = get_arg_val<uint32_t>(11);
    // This core's landing SLOT in its parent's reduce CBs (Refinement 2 lever 1) = its index in the
    // parent's child list modulo REDUCE_SLOTS, so the children of one invite wave never collide.
    const uint32_t my_slot = get_arg_val<uint32_t>(12);

    const auto wu_acc = TensorAccessor(wu_args, w_up_addr, BFP4_TILE);
    const auto out_acc = TensorAccessor(out_args, out_addr, BFP8_TILE);

    // The reader owns the device-resident count read and publishes it to the L1 mailbox.
    volatile tt_l1_ptr uint32_t* mbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_addr);
    while (mbox[moe_fused_swiglu::MBOX_READY] != MAILBOX_MAGIC) {
        invalidate_l1_cache();
    }
    const uint32_t m_t = mbox[moe_fused_swiglu::MBOX_M_T];
    const uint32_t m_blocks = mbox[moe_fused_swiglu::MBOX_M_BLOCKS];

    volatile tt_l1_ptr uint32_t* sem_go_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_GO)));
    uint32_t invites = 0;
#ifdef ABLATE_NO_REDUCE_XFER
    (void)sem_go_ptr;
    (void)invites;
    (void)parent_x;
    (void)parent_y;
    (void)my_slot;
#endif

    constexpr uint32_t SLOTS_H = REMAP ? (HID_T / NUM_BANKS) : HID_T;
    constexpr uint32_t SLOTS_E = REMAP ? (EMB_T / NUM_BANKS) : EMB_T;
    constexpr uint32_t WU_BLOCK_TILES = KR_PAD * HN_PAD;
    constexpr uint32_t SLOT_TILES = M_BLOCK * HN_PAD;  // one child's landing slot in the parent

    // The output block written but not yet barriered/popped — see the DEFERRED WRITE BARRIER below.
    uint32_t out_pending = 0;

    for (uint32_t b = 0; b < m_blocks; ++b) {
        // The RUNTIME token tile-rows this block works on — the SAME number the reader uses for its
        // multicast rounds and compute uses for its matmul shape (moe_fused_swiglu_common.hpp).
        const uint32_t m_eff = moe_fused_swiglu::m_tiles_eff(m_t, b, M_BLOCK, M_EFF_MIN);
        const uint32_t gu_block_tiles = m_eff * HN_PAD;
        const uint32_t out_block_tiles = m_eff * EC_MAX;

        // DEFERRED WRITE BARRIER (Refinement 2, the writer twin of the reader's deferred READ
        // barrier). The previous M-block's output write-back is drained HERE, not where it was
        // issued: `noc_async_write_barrier()` waits for every outstanding write, so barriering at
        // the issue site made the last stage of block `b` pay its full DRAM write latency with
        // nothing else in flight — and it sits between block `b` and block `b+1`, i.e. exactly on
        // the multi-M-block critical path (count > 256). Draining it here instead lets it ride this
        // block's W_up read. DEPTH_OUT >= 2 is what makes the extra outstanding block legal.
        if (out_pending) {
            noc_async_write_barrier();
            cb_pop_front(cb_out_tiles, out_pending);
            out_pending = 0;
        }

        // ---- W_up: NoC1 half of the gate/up weight stream, same bank-run coalescing ----
        cb_reserve_back(cb_w_up, WU_BLOCK_TILES);
        {
            MOE_ZONE("W_WUP");
            const uint32_t wp = get_write_ptr(cb_w_up);
#ifndef ABLATE_NO_W_XFER  // /perf-measure: drop the weight DRAM stream, keep every CB + barrier
            // REFINEMENT 3: M-block 0 only when W_up is resident (the read carries no `b`).
            if ((b == 0) || (W_RESIDENT == 0)) {
                for (uint32_t k = 0; k < kr; ++k) {
                    BR::read(
                        wu_acc,
                        (kstart + k) * HID_T,
                        hstart,
                        hstart + hn,
                        SLOTS_H,
                        wp + k * HN_PAD * BFP4_TILE,
                        BFP4_TILE);
                }
            }
#else
            (void)wp;
#endif
            noc_async_read_barrier();
        }
        cb_push_back(cb_w_up, WU_BLOCK_TILES);

        // ---- reduce tree, CHILD side ----
        if (!is_root) {
            MOE_ZONE("W_CHILD");
            cb_wait_front(cb_gate_send, gu_block_tiles);
            cb_wait_front(cb_up_send, gu_block_tiles);
#ifndef ABLATE_NO_REDUCE_XFER  // /perf-measure: drop the tree edge, keep the CB cycle
            // The parent invites us once per M-block; SEM_GO is monotone so no reset is needed.
            noc_semaphore_wait_min(sem_go_ptr, ++invites);
            // Every core has the identical CB layout and cb_reduce_*_in is pushed WHOLE (so its
            // write pointer is always the CB base on every core), so our own write pointer + our
            // slot stride IS the parent's landing address — no address negotiation. Only the m_eff
            // live tile-rows are shipped; the tail of the slot belongs to undefined token rows and
            // the parent drops it.
            const uint32_t slot_bytes = my_slot * SLOT_TILES * BFP8_TILE;
            noc_async_write(
                get_read_ptr(cb_gate_send),
                get_noc_addr(parent_x, parent_y, get_write_ptr(cb_reduce_gate_in) + slot_bytes),
                gu_block_tiles * BFP8_TILE);
            noc_async_write(
                get_read_ptr(cb_up_send),
                get_noc_addr(parent_x, parent_y, get_write_ptr(cb_reduce_up_in) + slot_bytes),
                gu_block_tiles * BFP8_TILE);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(parent_x, parent_y, static_cast<uint32_t>(get_semaphore(SEM_DATA))), 1);
#endif
            cb_pop_front(cb_gate_send, gu_block_tiles);
            cb_pop_front(cb_up_send, gu_block_tiles);
        }

        // ---- output write-back, coalesced over the emb axis ----
        // EC_MAX is the L1 row stride of the block (uniform CB increment); `ec` is how many of
        // those columns this core actually owns.
        {
            MOE_ZONE("W_OUT");
            cb_wait_front(cb_out_tiles, out_block_tiles);
            {
                const uint32_t rp = get_read_ptr(cb_out_tiles);
                for (uint32_t t = 0; t < m_eff; ++t) {
                    const uint32_t row = b * M_BLOCK + t;
                    if (row >= m_t) {
                        break;  // rows past ceil_tile(count) are never written
                    }
                    BR::write(
                        out_acc, row * EMB_T, jstart, jstart + ec, SLOTS_E, rp + t * EC_MAX * BFP8_TILE, BFP8_TILE);
                }
            }
            // Issued only — the barrier and the pop happen at the top of the NEXT M-block (or in the
            // epilogue below for the last one).
            out_pending = out_block_tiles;
        }
    }

    if (out_pending) {
        noc_async_write_barrier();
        cb_pop_front(cb_out_tiles, out_pending);
    }
}
