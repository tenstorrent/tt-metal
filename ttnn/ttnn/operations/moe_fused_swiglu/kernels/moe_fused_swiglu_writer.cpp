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
#include "moe_fused_swiglu_ct_args.hpp"    // the ONE definition of the compile-time arg order

MOE_DECLARE_CT_ENUM(MOE_WRITER_CT_ARGS);

constexpr uint32_t EMB_T = CT(EMB_T);
constexpr uint32_t HID_T = CT(HID_T);
constexpr uint32_t KR_PAD = CT(KR_PAD);
constexpr uint32_t HN_PAD = CT(HN_PAD);
constexpr uint32_t EC_MAX = CT(EC_MAX);  // phase-2 N stride (uniform CB increment)
constexpr uint32_t M_BLOCK = CT(M_BLOCK);
constexpr uint32_t HGROUPS = CT(HGROUPS);
constexpr uint32_t KGROUPS = CT(KGROUPS);

constexpr uint32_t SEM_GO = CT(SEM_GO);
constexpr uint32_t SEM_DATA = CT(SEM_DATA);
constexpr uint32_t SEM_HSLICE = CT(SEM_HSLICE);
constexpr uint32_t SEM_XSTAGED = CT(SEM_XSTAGED);
constexpr uint32_t SEM_WDSPLIT = CT(SEM_WDSPLIT);

constexpr uint32_t W_TILE = CT(W_TILE_BYTES);  // weight tile stride: bfp4 576, bfp8 1088, bf16 2048
constexpr uint32_t BFP8_TILE = CT(BFP8_TILE);
constexpr uint32_t H_TILE = BFP8_TILE;  // h is bfp8; see the reader
constexpr uint32_t MAILBOX_MAGIC = CT(MAILBOX_MAGIC);
constexpr uint32_t M_EFF_MIN = CT(M_EFF_MIN);
// Cross-M-block weight residency, the NoC1 half: W_up's read carries no M-block index, so every
// block after the first re-reads bytes still resident in cb_w_up's slot.
constexpr uint32_t W_RESIDENT = CT(W_RESIDENT);
constexpr uint32_t WD_RESIDENT = CT(WD_RESIDENT);
constexpr uint32_t GU_CHUNKS = CT(GU_CHUNKS);
constexpr uint32_t XPRIO = CT(XPRIO);
constexpr uint32_t WD_SPLIT = CT(WD_SPLIT);
constexpr uint32_t WG_SHARD_W = CT(WG_SHARD_W);
constexpr uint32_t WD_SHARD_W = CT(WD_SHARD_W);
// 1 moves the UP half of the gather to the reader (NOC_0), leaving GATE here on NOC_1. Split by
// PAYLOAD, not destination, so each RISC-V owns ONE CB outright: `cb_pop_front` writes the shared
// `tiles_acked` word with the popping RISC-V's own count, and two poppers corrupt it the way two
// pushers corrupt `tiles_received`.
constexpr uint32_t SCATTER_NOC_SPLIT = CT(SCATTER_NOC_SPLIT);

constexpr uint32_t cb_w_up = CT(CB_W_UP);
constexpr uint32_t cb_w_down = CT(CB_W_DOWN);
constexpr uint32_t cb_out_tiles = CT(CB_OUT_TILES);
constexpr uint32_t cb_gate_acc = CT(CB_GATE_ACC);
constexpr uint32_t cb_up_acc = CT(CB_UP_ACC);
constexpr uint32_t cb_gather_gate = CT(CB_GATHER_GATE);
constexpr uint32_t cb_gather_up = CT(CB_GATHER_UP);
constexpr uint32_t cb_h_slice = CT(CB_H_SLICE);
constexpr uint32_t cb_h_local = CT(CB_H_LOCAL);

// The accessor block follows the scalar block; CT_COUNT is its length.
constexpr auto wu_args = TensorAccessorArgs<CT_COUNT>();
constexpr auto out_args = TensorAccessorArgs<wu_args.next_compile_time_args_offset()>();
constexpr auto wd_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

// The W_down NoC split: WD_SPLIT eighths of every phase-2 K-block's hidden rows are read HERE on
// NOC_1 instead of on the reader's NOC_0, and published per K-block by transaction id.
constexpr uint32_t WD_BLOCK_TILES = HN_PAD * EC_MAX;  // the reader's twin — one phase-2 K-block
// The W_down stream takes its OWN tensor's DRAM ND shard width, exactly as the reader's `BRD` does.
using BRD = moe_fused_swiglu::WeightRuns<WD_SHARD_W>;

// Bank-run coalescing (see moe_fused_swiglu_bank_runs.hpp): ONE definition, bound here to this
// kernel's compile-time knobs — identical to the reader's binding, which is the point.
using BR = moe_fused_swiglu::WeightRuns<>;
// The W_up stream takes its tensor's DRAM ND shard width (0 = interleaved, byte-identical to the
// pre-WSHARD path). The OUTPUT write-back keeps `BR`: the output is always DRAM interleaved.
using BRG = moe_fused_swiglu::WeightRuns<WG_SHARD_W>;

// PER-STAGE ZONES — PERMANENT, always compiled, free with the profiler off (see the reader's note
// and the durability contract in `perf_instrumentation.hpp`). 5 records per M-block on either path:
// `tree` = out_drain, wup, reduce_child, out_issue; `scatter` = out_drain, wup, scatter, hslice,
// out_issue. Names of surviving stages are UNCHANGED so round-1 and round-2 numbers stay comparable.

void kernel_main() {
    const uint32_t mailbox_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_up_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t wd_addr = get_arg_val<uint32_t>(3);
    const uint32_t kr = get_arg_val<uint32_t>(4);
    const uint32_t kstart = get_arg_val<uint32_t>(5);
    const uint32_t hstart = get_arg_val<uint32_t>(6);
    const uint32_t hn = get_arg_val<uint32_t>(7);
    const uint32_t ec = get_arg_val<uint32_t>(8);
    const uint32_t jstart = get_arg_val<uint32_t>(9);
    const uint32_t my_col = get_arg_val<uint32_t>(10);
    // My row in the grid column. It IS my contributor slot in every peer's landing CB and it IS the
    // index of the slice I own, so the scatter needs no host-side plan table.
    const uint32_t my_row = get_arg_val<uint32_t>(11);
    // The row of THIS column's reduce root (`x % KGROUPS`) — the core the finished h slices are
    // gathered into, since it injects this column's h into the phase-2 all-gather.
    const uint32_t root_row = get_arg_val<uint32_t>(12);
    const bool is_root = (my_row == root_row);
    constexpr uint32_t RT_PEERS = 13;  // KGROUPS (vx, vy) pairs — the whole column, in row order

    const auto wu_acc = TensorAccessor(wu_args, w_up_addr, W_TILE);
    const auto out_acc = TensorAccessor(out_args, out_addr, BFP8_TILE);
    const auto wd_acc = TensorAccessor(wd_args, wd_addr, W_TILE);
    // THE ADDRESS DERIVATION, and why it needs no CB state. This RISC-V never pushes cb_w_down (the
    // reader is its single producer, which is the one-producer rule the split must not break), so
    // its local `cb_interface` copy never advances and `get_write_ptr` is the CB BASE for the whole
    // kernel. `WD_RESIDENT` forces the capacity to exactly HGROUPS K-blocks, so K-block r lives at
    // `base + r * WD_BLOCK_TILES * W_TILE` on every M-block. Must be read before anything else
    // touches the CB.
    const uint32_t wd_base = get_write_ptr(cb_w_down);

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
#endif
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
            if constexpr (XPRIO) {
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_XSTAGED)), b + 1);
            }
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
                            wp + k * GU_CHUNK_W * W_TILE,
                            W_TILE);
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
        auto issue_wd_share = [&]() {
            MaybeDeviceZoneScope("writer_wd_issue");
            // The publish word. A plain volatile store like SEM_XSTAGED: producer and consumer are
            // two RISC-Vs on the SAME core sharing one L1, and this word has exactly one writer.
            // It counts K-BLOCKS COMPLETED SINCE THE START OF THE OP, so it is monotone across
            // M-blocks and needs no reset — the same discipline as every other counter in this op.
            volatile tt_l1_ptr uint32_t* pub =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_WDSPLIT));
            if (!((b == 0) || (WD_RESIDENT == 0))) {
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
                // ONE TRANSACTION ID PER K-BLOCK. Every block still goes to DRAM at once (that
                // concurrency is the point of the batch); tagging lets the drain below release
                // block r when IT lands rather than when the last one does.
                noc_async_read_set_trid(r + 1);
                for (uint32_t k = hn_r - k_w; k < hn_r; ++k) {
                    // W_down's K axis is `h`'s hidden axis, so the row index goes through the
                    // same remap as the N axis — identical expression to the reader's.
                    BRD::read(
                        wd_acc,
                        (hbase + k) * EMB_T,
                        jstart,
                        jstart + ec,
                        wd_base + (r * WD_BLOCK_TILES + k * EC_MAX) * W_TILE,
                        W_TILE);
                }
            }
            noc_async_read_set_trid(0);  // back to untagged for the output write-back's cmd buf
            // DRAIN IN BLOCK ORDER, PUBLISHING AS WE GO. Costs the writer nothing over one blanket
            // barrier, but the reader stops waiting for the whole 111 KB stream and waits only for
            // the block it is about to push — worth +13 % at count 128.
            for (uint32_t r = 0; r < HGROUPS; ++r) {
                noc_async_read_barrier_with_trid(r + 1);
                *pub = b * HGROUPS + r + 1;
            }
        };
        issue_wd_share();

        // ---- REDUCE-SCATTER: contributor side + the finished-slice scatter ----
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
                const uint32_t w = i;
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
                    const uint32_t w = i;
                    const uint32_t vx = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 0);
                    const uint32_t vy = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 1);
                    noc_async_write(usrc + w * slice_bytes, get_noc_addr(vx, vy, udst + slot_bytes), slice_bytes);
                }
            }
            noc_async_write_barrier();
            const uint32_t sem_data = static_cast<uint32_t>(get_semaphore(SEM_DATA));
            for (uint32_t i = 0; i < sl_w; ++i) {
                const uint32_t w = i;
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
                    BR::write(out_acc, row * EMB_T, jstart, jstart + ec, rp + t * EC_MAX * BFP8_TILE, BFP8_TILE);
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
