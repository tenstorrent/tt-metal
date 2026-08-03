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
// MEASUREMENT KNOB (H_DTYPE) — the h intermediate's tile size in bytes. Deliberately SEPARATE
// from BFP8_TILE, which also sizes x, the output and the reduce operands: only the h transport
// follows this. Defaults to BFP8_TILE so the shipped path is byte-identical.
#ifndef H_TILE_BYTES
#define H_TILE_BYTES BFP8_TILE
#endif
constexpr uint32_t H_TILE = H_TILE_BYTES;

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
// PERF 17 (WD_SPLIT) — W_down's accessor, APPENDED LAST on the host so nothing above shifts.
constexpr auto wd_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

// PERF 17 — the W_down NoC split (NEXT_ROUND_PLAN #1 + #3, one lever). EIGHTHS of every phase-2
// K-block's hidden rows read HERE on NOC_1 instead of on the reader's NOC_0. 0 == the shipped
// all-reader stream and this whole block compiles away. Everything the split needs arrives as a
// DEFINE — `CB_W_DOWN_ID`, `WD_RESIDENT_D`, `SEM_WDSPLIT`, `WD_SPLIT` — because inserting a
// compile-time arg would shift TA_BASE in all three kernels (NEXT_ROUND_PLAN §4 trap 6).
#ifndef WD_SPLIT
#define WD_SPLIT 0
#endif
#ifndef WD_WPLACE_SCATTER
#define WD_WPLACE_SCATTER 0
#endif
// PERF 17b — publish granularity of the writer's share: 1 = one transaction id per K-block, so the
// reader is released block by block; 0 = one blanket barrier and one publish for the whole stream.
#ifndef WD_WPUB_TRID
#define WD_WPUB_TRID 0
#endif
// PERF 17b (NEXT_ROUND_PLAN §14.3) — drop the TRAILING atomic barrier after the gather's signals.
//
// Per M-block the gather costs TWO full ack round-trips: `noc_async_write_barrier()` (the real
// data-before-signal proof, which must stay) and then `noc_async_atomic_barrier()` after the
// semaphore increments. The second one guards nothing local — no later statement reads or reuses
// those cells, the payload is already proven landed, and the firmware's kernel-exit drain covers
// completion. 0 keeps the shipped barrier.
#ifndef SCATTER_NOBAR
#define SCATTER_NOBAR 0
#endif
#if SCATTER_NOBAR
#define MAYBE_ATOMIC_BARRIER() ((void)0)
#else
#define MAYBE_ATOMIC_BARRIER() noc_async_atomic_barrier()
#endif

// PERF 17b (NEXT_ROUND_PLAN §14.2) — ROTATE THE PEER-LOOP START.
//
// Every core walks its KGROUPS column peers from index 0 at the same instant, so all contributors
// hit peer 0 first, then peer 1, and so on: the reduce-scatter's all-to-all is issued as KGROUPS
// synchronised incasts rather than a spread. That is the shape tt-npe measured as **max link demand
// 302.6 %** against a 24.5 % average, and it is the signature the three gather stages carry as a
// 2.2x core-to-core spread (`writer_scatter` 36 350 ns mean / 59 979 max / 27 156 min).
//
// Starting core `r` at peer `r` and wrapping changes NOTHING but the order: the destination set, the
// per-peer source offsets and the transaction counts are identical, and every wait on the far side
// is a MONOTONE counter that cannot observe arrival order. 0 = the shipped in-order walk.
#ifndef SCATTER_ROT
#define SCATTER_ROT 0
#endif
//: Peer visited at iteration `i` of a `n`-long walk on the core in row `my_row`.
inline uint32_t peer_at(uint32_t i, uint32_t n, uint32_t my_row) {
#if SCATTER_ROT
    uint32_t w = i + my_row;
    while (w >= n) {
        w -= n;  // `my_row < KGROUPS` and `n <= KGROUPS`, so at most one wrap past the first
    }
    return w;
#else
    (void)my_row;
    (void)n;
    return i;
#endif
}

// PERF 17 (HSPLIT) — this RISC-V broadcasts the TAIL tiles of every h all-gather round on NOC_1
// while the reader keeps the head tiles (and the linked VALID flag) on NOC_0. See
// MOE_SWIGLU_HSPLIT in the descriptor for the completion protocol. 0 == the shipped reader-only
// multicast and this whole block compiles away.
#ifndef HSPLIT
#define HSPLIT 0
#endif
//: The TAIL tiles this RISC-V broadcasts — the exact complement of the reader's `h_tiles_noc0`, so
//: the two halves tile the block with no gap and no overlap at every runtime m_eff.
inline uint32_t h_tiles_noc1(uint32_t t) { return (t * HSPLIT) / 8; }

#if WD_SPLIT
constexpr uint32_t cb_w_down = CB_W_DOWN_ID;
constexpr uint32_t WD_BLOCK_TILES = HN_PAD * EC_MAX;  // the reader's twin — one phase-2 K-block
// The W_down stream takes its OWN tensor's DRAM ND shard width, exactly as the reader's `BRD` does.
using BRD = moe_fused_swiglu::BankRuns<REMAP != 0, NUM_BANKS, WRUN, WD_SHARD_W>;
#endif

// Bank-run coalescing (see moe_fused_swiglu_bank_runs.hpp): ONE definition, bound here to this
// kernel's compile-time knobs — identical to the reader's binding, which is the point.
using BR = moe_fused_swiglu::BankRuns<REMAP != 0, NUM_BANKS, WRUN>;
// The W_up stream takes its tensor's DRAM ND shard width (0 = interleaved, byte-identical to the
// pre-WSHARD path). The OUTPUT write-back keeps `BR`: the output is always DRAM interleaved.
using BRG = moe_fused_swiglu::BankRuns<REMAP != 0, NUM_BANKS, WRUN, WG_SHARD_W>;

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
    constexpr uint32_t RT_WDADDR = RT_MYCOL + 1;  // PERF 17 (WD_SPLIT) — W_down's base address

    const auto wu_acc = TensorAccessor(wu_args, w_up_addr, BFP4_TILE);
    const auto out_acc = TensorAccessor(out_args, out_addr, BFP8_TILE);
#if WD_SPLIT
    const auto wd_acc = TensorAccessor(wd_args, get_arg_val<uint32_t>(RT_WDADDR), BFP4_TILE);
    // THE ADDRESS DERIVATION, and why it needs no CB state. This RISC-V never pushes cb_w_down (the
    // reader is its single producer, which is the one-producer rule the split must not break), so its
    // local `cb_interface` copy never advances and `get_write_ptr` is the CB BASE for the whole
    // kernel. `WD_RESIDENT` forces the capacity to exactly HGROUPS K-blocks, so K-block r lives at
    // `base + r * WD_BLOCK_TILES * BFP4_TILE` on every M-block. Same derivation HSEND=writer uses for
    // `cb_h_base`, and it must likewise be read before anything else touches the CB.
    const uint32_t wd_base = get_write_ptr(cb_w_down);
#endif

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
#if HSEND_WRITER || HSPLIT
    // PERF 3 / PERF 17 — the h broadcast's two running totals, mirroring the reader's discipline:
    // monotone, never reset, always compared with wait_min. `cb_h_base` MUST be read before any
    // push. This RISC-V never pushes cb_h (the reader is its single producer), so its `cb_interface`
    // copy never advances and this stays the CB base for the whole kernel.
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
                        BRG::read(
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

        // ---- PERF 17: MY SHARE OF THE PHASE-2 W_down STREAM, on NOC_1 ----
        //
        // WD_SPLIT eighths of every K-block's hidden rows are read here so they do not compete with
        // the h all-gather for the reader's NOC_0 request path (NEXT_ROUND_PLAN #1/#3). ALL HGROUPS
        // blocks go out as one batch: with `WD_RESIDENT` every W_down DRAM read happens at b == 0,
        // where all HGROUPS slots are free from kernel start, so there is nothing to flow-control
        // against and the whole stream can be in flight at once. The one thing that IS mandatory is
        // the completion handshake: `noc_async_read_barrier()` is per-RISC-V, so the reader's barrier
        // proves nothing about these reads.
        //
        // The ROWS are split, not the blocks, and this RISC-V takes the TAIL rows `[hn_r - k_w, hn_r)`
        // — a contiguous run, so the bank-run coalescing is unaffected on either side.
#if WD_SPLIT
        auto issue_wd_share = [&]() {
            MaybeDeviceZoneScope("writer_wd_issue");
            // The publish word. A plain volatile store like SEM_XSTAGED: producer and consumer are
            // two RISC-Vs on the SAME core sharing one L1, and this word has exactly one writer.
            // It counts K-BLOCKS COMPLETED SINCE THE START OF THE OP, so it is monotone across
            // M-blocks and needs no reset — the same discipline as every other counter in this op.
            volatile tt_l1_ptr uint32_t* pub =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_WDSPLIT));
            if (!((b == 0) || (WD_RESIDENT_D == 0))) {
                *pub = (b + 1) * HGROUPS;  // resident: nothing to read, the bytes are already there
                return;
            }
            for (uint32_t r = 0; r < HGROUPS; ++r) {
                const uint32_t hbase = r * HN_PAD;
                uint32_t hn_r = HN_PAD;
                if (hbase + hn_r > HID_T) {
                    hn_r = HID_T - hbase;
                }
                const uint32_t k_w = (hn_r * WD_SPLIT) / 8;  // rows read HERE
#if WD_WPUB_TRID
                // PERF 17b — ONE TRANSACTION ID PER K-BLOCK. Every block still goes to DRAM at once
                // (that concurrency is the whole point of the batch), but tagging them lets the
                // drain below release block r the moment IT lands instead of when the LAST one does.
                noc_async_read_set_trid(r + 1);
#endif
                for (uint32_t k = hn_r - k_w; k < hn_r; ++k) {
                    // W_down's K axis is `h`'s hidden axis, so the row index goes through the
                    // same remap as the N axis — identical expression to the reader's.
                    BRD::read(
                        wd_acc,
                        BRG::remap(hbase + k, SLOTS_H) * EMB_T,
                        jstart,
                        jstart + ec,
                        SLOTS_E,
                        wd_base + (r * WD_BLOCK_TILES + k * EC_MAX) * BFP4_TILE,
                        BFP4_TILE);
                }
            }
#if WD_WPUB_TRID
            noc_async_read_set_trid(0);  // back to untagged for the output write-back's cmd buf
            // DRAIN IN BLOCK ORDER, PUBLISHING AS WE GO. This costs the writer nothing over the
            // single blanket barrier — the last `barrier_with_trid` returns at the same instant a
            // whole-batch barrier would — but the READER stops waiting for the whole 111 KB stream
            // and starts waiting only for the block it is about to push. That distinction is the
            // difference between WD_SPLIT being free at count 128 and costing +13 % there.
            for (uint32_t r = 0; r < HGROUPS; ++r) {
                noc_async_read_barrier_with_trid(r + 1);
                *pub = b * HGROUPS + r + 1;
            }
#else
            noc_async_read_barrier();
            *pub = (b + 1) * HGROUPS;
#endif
        };
#if !WD_WPLACE_SCATTER
        issue_wd_share();
#endif
#endif

        // ---- PERF 2: REDUCE-SCATTER, contributor side + the finished-slice scatter ----
        if constexpr (SCATTER) {
            // The ONE shared slice plan (moe_fused_swiglu_common.hpp), from the SAME (m_eff, KGROUPS)
            // compute and the reader use. `sl_w` workers own `sl_a` tiles each; rows >= sl_w are idle
            // for the reduce but still CONTRIBUTE, which is why the send loop below is unconditional.
            const uint32_t sl_w = moe_fused_swiglu::slice_workers(gu_block_tiles, KGROUPS);
            const uint32_t sl_a = gu_block_tiles / sl_w;
            const uint32_t slice_bytes = sl_a * H_TILE;
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
                for (uint32_t i = 0; i < sl_w; ++i) {
                    const uint32_t w = peer_at(i, sl_w, my_row);  // §14.2 rotation
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
                    for (uint32_t i = 0; i < sl_w; ++i) {
                        const uint32_t w = peer_at(i, sl_w, my_row);  // §14.2 rotation
                        const uint32_t vx = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 0);
                        const uint32_t vy = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 1);
                        noc_async_write(usrc + w * slice_bytes, get_noc_addr(vx, vy, udst + slot_bytes), slice_bytes);
                    }
                }
                noc_async_write_barrier();
                const uint32_t sem_data = static_cast<uint32_t>(get_semaphore(SEM_DATA));
                for (uint32_t i = 0; i < sl_w; ++i) {
                    const uint32_t w = peer_at(i, sl_w, my_row);  // §14.2 rotation
                    const uint32_t vx = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 0);
                    const uint32_t vy = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 1);
                    noc_semaphore_inc(get_noc_addr(vx, vy, sem_data), 1);
                }
                MAYBE_ATOMIC_BARRIER();
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
                MAYBE_ATOMIC_BARRIER();
#endif
                cb_pop_front(cb_h_slice, sl_a);
            }

#if HSPLIT
            // ---- PERF 17: the NOC_1 HALF of this column's h broadcast ----
            //
            // The reader owns the head tiles and the round's linked VALID flag; this owns the tail
            // tiles and its own monotone per-slot arrival counter. `cb_h`'s reserve/push stays
            // entirely on the reader — this is a raw NoC write into the region the reader already
            // reserved, so the CB keeps exactly ONE producer (§4 trap 5).
            if (is_root) {
                MaybeDeviceZoneScope("writer_hsplit");
                const uint32_t n1 = h_tiles_noc1(gu_block_tiles);
                // (a) my column's h block is assembled — the same monotone counter the reader waits
                //     on, and a WAIT (not a consume), so both RISC-Vs may read it.
                h_arrivals += sl_w;
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_HSLICE))),
                    h_arrivals);
                // (b) every core has reserved the slot this round lands in. ONE monotone counter
                //     shared with the reader's own send: a receiver acks the SENDING CORE once per
                //     round, so the cell gains exactly NUM_CORES per M-block whichever RISC-V reads
                //     it, and both keep their own identical running expectation.
                hfree_expected += NUM_CORES;
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_H_FREE))),
                    hfree_expected);
#ifndef ABLATE_NO_H_XFER
                if (n1) {
                    // Landing slot, derived exactly as HSEND=writer derives it: every core pushes
                    // cb_h in lockstep in units of `gu_block_tiles`, so the slot index is
                    // (b*HGROUPS + my_col) mod (capacity / block) off the base captured above.
                    const uint32_t blocks_cap = (DEPTH_H * M_BLOCK) / m_eff;
                    const uint32_t slot = (b * HGROUPS + my_col) % blocks_cap;
                    const uint32_t off = (gu_block_tiles - n1) * H_TILE;  // the reader's head, skipped
                    const uint64_t dst = get_noc_multicast_addr(
                        get_arg_val<uint32_t>(RT_HRECT + 0),
                        get_arg_val<uint32_t>(RT_HRECT + 1),
                        get_arg_val<uint32_t>(RT_HRECT + 2),
                        get_arg_val<uint32_t>(RT_HRECT + 3),
                        cb_h_base + slot * gu_block_tiles * H_TILE + off);
                    // EXCLUDE-source (fan-out NUM_CORES - 1): this core already holds the WHOLE
                    // block, because the reader's self-copy lands all of it — head and tail — in
                    // the local cb_h slot before either half goes out.
                    noc_async_write_multicast(
                        get_write_ptr(cb_h_local) + off, dst, n1 * H_TILE, NUM_CORES - 1, /*linked=*/false);
                    // ACKED, not merely flushed. Unlike the reader's half there is no LINK to carry
                    // the ordering: the counter below rides a different command buffer and cannot
                    // terminate a NOC_CMD_VC_LINKED chain (the Perf-3 addendum-2 finding), so the
                    // payload must have LANDED before the signal moves.
                    noc_async_write_barrier();
                    noc_semaphore_inc_multicast(
                        get_noc_multicast_addr(
                            get_arg_val<uint32_t>(RT_HRECT + 0),
                            get_arg_val<uint32_t>(RT_HRECT + 1),
                            get_arg_val<uint32_t>(RT_HRECT + 2),
                            get_arg_val<uint32_t>(RT_HRECT + 3),
                            static_cast<uint32_t>(get_semaphore(SEM_H2_RDY_BASE + ((b * HGROUPS + my_col) % DEPTH_H)))),
                        1,
                        NUM_CORES - 1);
                    MAYBE_ATOMIC_BARRIER();
                }
#endif
            }
#endif

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
                const uint32_t hbytes = gu_block_tiles * H_TILE;
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
                MAYBE_ATOMIC_BARRIER();
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
            const uint32_t slot_bytes = my_slot * SLOT_TILES * H_TILE;
            noc_async_write(
                get_read_ptr(cb_gate_send),
                get_noc_addr(parent_x, parent_y, get_write_ptr(cb_reduce_gate_in) + slot_bytes),
                gu_block_tiles * H_TILE);
            noc_async_write(
                get_read_ptr(cb_up_send),
                get_noc_addr(parent_x, parent_y, get_write_ptr(cb_reduce_up_in) + slot_bytes),
                gu_block_tiles * H_TILE);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(parent_x, parent_y, static_cast<uint32_t>(get_semaphore(SEM_DATA))), 1);
#endif
            cb_pop_front(cb_gate_send, gu_block_tiles);
            cb_pop_front(cb_up_send, gu_block_tiles);
        }

#if WD_SPLIT && WD_WPLACE_SCATTER
        // PERF 17 — the `scatter` placement: past the column all-to-all, so these reads never share
        // NOC_1 with the gather's own writes. The writer twin of WD_LATE. Placed after the whole
        // reduce if/else so it is reached on the `tree` path too.
        issue_wd_share();
#endif

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
