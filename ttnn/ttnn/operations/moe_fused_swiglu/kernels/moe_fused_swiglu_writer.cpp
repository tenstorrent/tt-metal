// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — WRITER (NoC1).
//
// Owns, per M-block:
//   1. the W_up weight stream — the NoC1 twin of the reader's W_gate stream (op_design.md §1.5
//      dual-issue split: a phase with two independent weight streams uses BOTH data-movement
//      RISC-Vs / both NoCs);
//   2. this core's send side of the gate/up cross-column reduce, in ONE OF TWO SHAPES
//      (MOE_SWIGLU_REDUCE):
//        `tree`    — the CHILD side of the binary tree: wait for the parent's invite, unicast both
//                    whole partials into the parent's cb_reduce_*_in, signal;
//        `scatter` — PERF 2, the default: wait for the whole column's invites, then unicast MY SLICE
//                    of gate (and of up, unless MOE_SWIGLU_SCATTER_NOC=split moved that half to the
//                    reader's NoC) into every worker's landing CB, signal each; and, once compute has
//                    finished my slice's epilogue, unicast the finished `h` slice straight into the
//                    column ROOT's cb_h_local at its tile offset — the gather IS the assembly;
//   3. the coalesced bank-run output write-back, clamped to tile-rows < ceil_tile(count) so rows
//      past the real token count are never touched.
//
// Raw-dataflow deviations are the same two documented at the head of the reader: bank-run
// noc_async_write/read for coalescing (no in-tree helper expresses a multi-page contiguous
// transaction on an interleaved tensor), and raw unicast + counting semaphores for the tree edge
// (mcast_pipe's SenderPipe is a rectangle multicast, not a point-to-point tree edge).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

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

// PERF 2 — REDUCE-SCATTER (MOE_SWIGLU_REDUCE=scatter). 1 replaces the tree's single unicast-to-parent
// with the column all-to-all plus the finished-h-slice scatter to the root; 0 reproduces the tree
// byte-for-byte. See the knob in the program descriptor for the measurement and the predicate.
constexpr uint32_t SCATTER = get_compile_time_arg_val(25);
// 1 moves the UP half of the gather to the reader (NOC_0), leaving the GATE half here on NOC_1 — see
// MOE_SWIGLU_SCATTER_NOC. Split by PAYLOAD, not by destination, because each RISC-V then owns ONE
// CB outright: `cb_pop_front` writes the shared `tiles_acked` word with the popping RISC-V's own
// local count, so two poppers on one CB is the same silent-corruption class as two pushers.
constexpr uint32_t SCATTER_NOC_SPLIT = get_compile_time_arg_val(26);
constexpr uint32_t SEM_HSLICE = get_compile_time_arg_val(27);
constexpr uint32_t cb_gate_acc = get_compile_time_arg_val(28);
constexpr uint32_t cb_up_acc = get_compile_time_arg_val(29);
constexpr uint32_t cb_gather_gate = get_compile_time_arg_val(30);
constexpr uint32_t cb_gather_up = get_compile_time_arg_val(31);
constexpr uint32_t cb_h_slice = get_compile_time_arg_val(32);
constexpr uint32_t cb_h_local = get_compile_time_arg_val(33);
// PERF 3 (HSEND=writer) — the h all-gather CB. The writer never pushes it; it needs only its
// BASE address, captured before any push, to derive each round's landing slot arithmetically.
constexpr uint32_t cb_h = get_compile_time_arg_val(34);

constexpr uint32_t TA_BASE = 35;
constexpr auto wu_args = TensorAccessorArgs<TA_BASE>();
constexpr auto out_args = TensorAccessorArgs<wu_args.next_compile_time_args_offset()>();

// Bank-run coalescing (see moe_fused_swiglu_bank_runs.hpp): ONE definition, bound here to this
// kernel's compile-time knobs — identical to the reader's binding, which is the point.
using BR = moe_fused_swiglu::BankRuns<REMAP != 0, NUM_BANKS, WRUN>;

// PER-STAGE ZONES — PERMANENT, always compiled, free with the profiler off (see the reader's note
// and the durability contract in `perf_instrumentation.hpp`). 5 records per M-block on either path:
// `tree` = out_drain, wup, reduce_child, out_issue; `scatter` = out_drain, wup, scatter, hslice,
// out_issue. Names of surviving stages are UNCHANGED so round-1 and round-2 numbers stay comparable.

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
    // PERF 2 — my row in the grid column. It IS my contributor slot in every peer's landing CB and it
    // IS the index of the slice I own, so the scatter needs no host-side plan table.
    const uint32_t my_row = get_arg_val<uint32_t>(13);
    // The row of THIS column's reduce root (`x % KGROUPS`) — the one core the finished h slices are
    // gathered into, since it is the core that injects this column's h into the phase-2 all-gather.
    const uint32_t root_row = get_arg_val<uint32_t>(14);
    constexpr uint32_t RT_PEERS = 15;  // KGROUPS (vx, vy) pairs — the whole column, in row order
    // PERF 3 (HSEND=writer) — the h broadcast rect, NOC1 routing order: (far_x, far_y, near_x, near_y).
    constexpr uint32_t RT_HRECT = RT_PEERS + 2 * KGROUPS;
    constexpr uint32_t RT_MYCOL = RT_HRECT + 4;  // this core's grid COLUMN == the round it sends

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
#if HSEND_WRITER
    // PERF 3 — the h broadcast's two running totals, mirroring the reader's discipline: monotone,
    // never reset, always compared with wait_min. `cb_h_base` MUST be read before any push.
    const uint32_t my_col = get_arg_val<uint32_t>(RT_MYCOL);
    const uint32_t cb_h_base = get_write_ptr(cb_h);
    uint32_t h_arrivals = 0;
    uint32_t hfree_expected = 0;
#endif
#ifdef ABLATE_NO_REDUCE_XFER
    (void)sem_go_ptr;
    (void)invites;
    (void)parent_x;
    (void)parent_y;
    (void)my_slot;
    (void)root_row;
#endif

    constexpr uint32_t SLOTS_H = REMAP ? (HID_T / NUM_BANKS) : HID_T;
    constexpr uint32_t SLOTS_E = REMAP ? (EMB_T / NUM_BANKS) : EMB_T;
    constexpr uint32_t WU_BLOCK_TILES = KR_PAD * HN_PAD;
    // PERF 3 — the N-chunk the weight stream is published in (reader's twin). 1 == the whole block.
    constexpr uint32_t GU_CHUNK_W = HN_PAD / GU_CHUNKS;
    constexpr uint32_t WU_CHUNK_TILES = KR_PAD * GU_CHUNK_W;
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
            MaybeDeviceZoneScope("writer_out_drain");
            noc_async_write_barrier();
            cb_pop_front(cb_out_tiles, out_pending);
            out_pending = 0;
        }

        // ---- W_up: NoC1 half of the gate/up weight stream, same bank-run coalescing ----
        //
        // PERF 3 — published in the same `GU_CHUNKS` N-chunks as the reader's W_gate twin, for the
        // same reason: the up matmul starts on chunk 0 while chunk 1 is still in DRAM. See the
        // reader's comment for why N (independent chunks, no extra accumulating pack) and not K.
        {
            MaybeDeviceZoneScope("writer_wup");
            // PERF 3 — ACTIVATION-FIRST. Hold this NoC1 weight stream until this core's reader has
            // pulled its `x` tile-rows off DRAM. Every core does it, so the whole grid's 16.5 MB of
            // gate/up weights stays behind the 3.67 MB of activation that EVERY core's matmul is
            // blocked on. Intra-core: an L1 poll, no NoC traffic. Never entered when m_blocks == 0,
            // so the zero-count dispatch still cannot hang.
#if XPRIO
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_XSTAGED)), b + 1);
#endif
            for (uint32_t c = 0; c < GU_CHUNKS; ++c) {
                cb_reserve_back(cb_w_up, WU_CHUNK_TILES);
                const uint32_t wp = get_write_ptr(cb_w_up);
                const uint32_t h0 = c * GU_CHUNK_W;
                uint32_t w = (h0 < hn) ? (hn - h0) : 0;
                if (w > GU_CHUNK_W) {
                    w = GU_CHUNK_W;
                }
#ifndef ABLATE_NO_W_XFER  // /perf-measure: drop the weight DRAM stream, keep every CB + barrier
                // REFINEMENT 3: M-block 0 only when W_up is resident (the read carries no `b`).
                if (((b == 0) || (W_RESIDENT == 0)) && w) {
                    for (uint32_t k = 0; k < kr; ++k) {
                        BR::read(
                            wu_acc,
                            (kstart + k) * HID_T,
                            hstart + h0,
                            hstart + h0 + w,
                            SLOTS_H,
                            wp + k * GU_CHUNK_W * BFP4_TILE,
                            BFP4_TILE);
                    }
                }
#else
                (void)wp;
#endif
                noc_async_read_barrier();
                cb_push_back(cb_w_up, WU_CHUNK_TILES);
            }
        }

        // ---- PERF 2: REDUCE-SCATTER, contributor side + the finished-slice scatter ----
        if constexpr (SCATTER) {
            // The ONE shared slice plan (moe_fused_swiglu_common.hpp), from the SAME (m_eff, KGROUPS)
            // compute and the reader use. `sl_w` workers own `sl_a` tiles each; rows >= sl_w are idle
            // for the reduce but still CONTRIBUTE, which is why the send loop below is unconditional.
            const uint32_t sl_w = moe_fused_swiglu::slice_workers(gu_block_tiles, KGROUPS);
            const uint32_t sl_a = gu_block_tiles / sl_w;
            const uint32_t slice_bytes = sl_a * BFP8_TILE;
            {
                MaybeDeviceZoneScope("writer_scatter");
#ifndef ABLATE_NO_REDUCE_XFER  // /perf-measure: drop the all-to-all, keep every CB cycle
                // KGROUPS invites per M-block is the generalisation of the tree's one-parent SEM_GO:
                // every peer's reader reserves its landing CBs and THEN invites the whole column, so
                // this wait is what stops block b+1's contribution from overwriting a landing slot
                // compute has not consumed yet. MONOTONE, never reset — the running total only grows.
                noc_semaphore_wait_min(sem_go_ptr, invites + KGROUPS);
#endif
                invites += KGROUPS;
                cb_wait_front(cb_gate_acc, gu_block_tiles);
                if constexpr (SCATTER_NOC_SPLIT == 0) {
                    cb_wait_front(cb_up_acc, gu_block_tiles);
                }
#ifndef ABLATE_NO_REDUCE_XFER
                const uint32_t gsrc = get_read_ptr(cb_gate_acc);
                // LANDING ADDRESS = MY OWN write pointer + MY slot. Every core has the identical CB
                // layout and the landing CBs are pushed WHOLE every M-block, so my write pointer is
                // the CB base on the destination too — the same address proxy the tree edge uses,
                // with KGROUPS disjoint slots instead of one parent's slot.
                const uint32_t gdst = get_write_ptr(cb_gather_gate);
                const uint32_t slot_bytes = my_row * slice_bytes;
                for (uint32_t w = 0; w < sl_w; ++w) {
                    const uint32_t vx = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 0);
                    const uint32_t vy = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 1);
                    // ONE coalesced transaction per leg: worker `w` owns the CONTIGUOUS tile range
                    // [w*sl_a, (w+1)*sl_a) because the gate/up block layout is `m*HN_PAD + n`
                    // (OUT_SUBBLOCK_H_GU == 1, SubblockMajor). A token-axis slice would be strided.
                    noc_async_write(gsrc + w * slice_bytes, get_noc_addr(vx, vy, gdst + slot_bytes), slice_bytes);
                }
                if constexpr (SCATTER_NOC_SPLIT == 0) {
                    const uint32_t usrc = get_read_ptr(cb_up_acc);
                    const uint32_t udst = get_write_ptr(cb_gather_up);
                    for (uint32_t w = 0; w < sl_w; ++w) {
                        const uint32_t vx = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 0);
                        const uint32_t vy = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 1);
                        noc_async_write(usrc + w * slice_bytes, get_noc_addr(vx, vy, udst + slot_bytes), slice_bytes);
                    }
                }
                noc_async_write_barrier();
                const uint32_t sem_data = static_cast<uint32_t>(get_semaphore(SEM_DATA));
                for (uint32_t w = 0; w < sl_w; ++w) {
                    const uint32_t vx = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 0);
                    const uint32_t vy = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 1);
                    noc_semaphore_inc(get_noc_addr(vx, vy, sem_data), 1);
                }
                noc_async_atomic_barrier();
#endif
                cb_pop_front(cb_gate_acc, gu_block_tiles);
                if constexpr (SCATTER_NOC_SPLIT == 0) {
                    cb_pop_front(cb_up_acc, gu_block_tiles);
                }
            }
            // ---- my finished h slice, straight into the ROOT's cb_h_local at its tile offset ----
            // The gather IS the assembly. cb_h_local is never pushed or popped on this path, so its
            // write pointer is the CB base on every core for every M-block — which is what lets this
            // core use its OWN pointer as the root's. Flow control across M-blocks is the invite
            // above, transitively: my send here is downstream of the gather, the gather is downstream
            // of the ROOT's invite for this block, and the root only invites after its phase 2 of the
            // PREVIOUS block has read cb_h_local (and barriered).
            if (my_row < sl_w) {
                MaybeDeviceZoneScope("writer_hslice");
                cb_wait_front(cb_h_slice, sl_a);
#ifndef ABLATE_NO_REDUCE_XFER
                const uint32_t rvx = get_arg_val<uint32_t>(RT_PEERS + 2 * root_row + 0);
                const uint32_t rvy = get_arg_val<uint32_t>(RT_PEERS + 2 * root_row + 1);
                noc_async_write(
                    get_read_ptr(cb_h_slice),
                    get_noc_addr(rvx, rvy, get_write_ptr(cb_h_local) + my_row * slice_bytes),
                    slice_bytes);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(rvx, rvy, static_cast<uint32_t>(get_semaphore(SEM_HSLICE))), 1);
                noc_async_atomic_barrier();
#endif
                cb_pop_front(cb_h_slice, sl_a);
            }

#if HSEND_WRITER
            // ---- PERF 3: the h BROADCAST, moved off the reader ----
            //
            // THE POINT OF THE SPLIT. On the reader this send sat at iteration `r == my_col` of the
            // same loop that receives every other column, so the sender of round r+1 could not start
            // until it had RECEIVED rounds 0..r — an HGROUPS-long serial chain that measured as the
            // whole of phase 2 (43 us ~= 11 x 3.9 us with every payload ablated). Here it is on a
            // RISC-V that is not in that loop, so a root broadcasts as soon as (a) its column's h is
            // assembled and (b) the grid has freed its slot. `consume(r)` then depends only on
            // `consume(r - DEPTH_H)`, i.e. a chain of HGROUPS/DEPTH_H, and it costs NO extra L1
            // because the rolling window is unchanged.
            if (is_root) {
                MaybeDeviceZoneScope("writer_hsend");
                h_arrivals += sl_w;
                // (a) my column's h block is complete — the same monotone counter the reader waits
                //     on, and a wait (not a consume), so both RISC-Vs may read it.
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_HSLICE))),
                    h_arrivals);
                // (b) every core has reserved the slot this round lands in. One ack per core per
                //     round; the receiver sends it BEFORE waiting for data, which is what makes the
                //     split deadlock-free.
                hfree_expected += NUM_CORES;
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_H_FREE))),
                    hfree_expected);
#ifndef ABLATE_NO_H_XFER
                // The landing address is NOT `get_write_ptr(cb_h)` — `cb_interface` is per-RISC-V
                // local state and this RISC-V is not cb_h's pusher, so its copy never advances.
                // It is derived instead: every core pushes cb_h in lockstep in units of
                // `h_block_tiles`, so slot index = (b * HGROUPS + my_col) mod (capacity / block),
                // measured from the base captured before any push.
                const uint32_t blocks_cap = (DEPTH_H * M_BLOCK) / m_eff;
                const uint32_t slot = (b * HGROUPS + my_col) % blocks_cap;
                // `gu_block_tiles` IS the reader's `h_block_tiles`: both are m_eff * HN_PAD.
                const uint32_t hbytes = gu_block_tiles * BFP8_TILE;
                const uint64_t dst = get_noc_multicast_addr(
                    get_arg_val<uint32_t>(RT_HRECT + 0),
                    get_arg_val<uint32_t>(RT_HRECT + 1),
                    get_arg_val<uint32_t>(RT_HRECT + 2),
                    get_arg_val<uint32_t>(RT_HRECT + 3),
                    cb_h_base + slot * hbytes);
                // EXCLUDE-source: this core's own copy is the reader's local self-copy, so the
                // multicast fan-out is NUM_CORES - 1 and this core's own SEM_H_RDY is not bumped —
                // which is exactly why the reader keeps a private per-slot expectation.
                noc_async_write_multicast(get_write_ptr(cb_h_local), dst, hbytes, NUM_CORES - 1, /*linked=*/false);
                // ACKED, not merely flushed: the receivers key on the counter below, so the payload
                // must have LANDED before it moves. Same rule the mcast_pipe Counter fix follows.
                noc_async_write_barrier();
                noc_semaphore_inc_multicast(
                    get_noc_multicast_addr(
                        get_arg_val<uint32_t>(RT_HRECT + 0),
                        get_arg_val<uint32_t>(RT_HRECT + 1),
                        get_arg_val<uint32_t>(RT_HRECT + 2),
                        get_arg_val<uint32_t>(RT_HRECT + 3),
                        static_cast<uint32_t>(get_semaphore(SEM_H_RDY_BASE + (my_col % DEPTH_H)))),
                    1,
                    NUM_CORES - 1);
                noc_async_atomic_barrier();
#endif
            }
#endif
        } else if (!is_root) {
            // ---- reduce tree, CHILD side ----
            MaybeDeviceZoneScope("writer_reduce_child");
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
            MaybeDeviceZoneScope("writer_out_issue");
            cb_wait_front(cb_out_tiles, out_block_tiles);
#ifndef ABLATE_NO_OWRITE  // /perf-measure: drop the output DRAM bytes, keep the CB cycle + barrier
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
#endif
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
