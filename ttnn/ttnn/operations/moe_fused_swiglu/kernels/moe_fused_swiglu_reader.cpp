// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — READER (NoC0).
//
// Owns, per M-block:
//   1. the device-resident token count read (idx -> counts[idx[local_expert_id]]), published to
//      the L1 mailbox every kernel spins on;
//   2. the `x` activation stage + the ROW-multicast of x along grid row y (rotating injector,
//      one tile-row per round) via mcast_pipe;
//   3. the coalesced bank-run W_gate stream (the W_up twin lives on the writer/NoC1);
//   4. the RECEIVER side of the gate/up cross-column reduce, in ONE OF TWO SHAPES
//      (MOE_SWIGLU_REDUCE):
//        `tree`    — the PARENT side of the binary tree: invite child -> land its whole partials in
//                    cb_reduce_*_in -> publish to compute;
//        `scatter` — PERF 2, the default: reserve my two landing CBs, invite the WHOLE COLUMN, wait
//                    for every contributor's slice, publish. On a column ROOT it additionally waits
//                    on SEM_HSLICE instead of a cb_h_local front, because on this path the workers'
//                    NoC writes ARE what assembles cb_h_local;
//   5. the phase-2 loop: one coalesced W_down K-block + one round of the grid-wide `h`
//      all-gather per iteration.
//
// RAW-DATAFLOW DEVIATIONS (documented per op_design.md §6 "Helpers considered and rejected"):
//   * Weight/activation/output DRAM traffic uses raw `noc_async_read` with a bank-contiguous RUN
//     length instead of the page-granular TensorAccessor read helper: the page-granular helper
//     issues one transaction per 576 B tile, which is transaction-rate-bound (~5 GB/s/core). No
//     in-tree helper expresses "read L bank-contiguous pages as one transaction" for an
//     INTERLEAVED tensor (get_max_page_size_and_num_pages is DRAM-sharded-only). Address
//     computation still goes through TensorAccessor. `WRUN = 1` reproduces the naive per-tile read.
//   * The reduce-tree transport is raw unicast + counting semaphores rather than mcast_pipe:
//     SenderPipe is a rectangle MULTICAST sender, while a tree edge is point-to-point with a
//     different destination per node and level, and a tree node needs counting fan-in.
//     mcast_pipe IS used for both real broadcasts (x row-mcast, h all-gather).
//   * The token-count publish is a raw L1 mailbox because the compute kernel needs a scalar loop
//     bound on ALL THREE TRISCs and `cb_wait_front` in a compute kernel is UNPACK-only.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#include "moe_fused_swiglu_bank_runs.hpp"  // the ONE definition of the bank-run coalescing
#include "moe_fused_swiglu_common.hpp"     // the ONE definition of the mailbox word layout

using namespace dataflow_kernel_lib;

// PER-STAGE ZONES — PERMANENT. Each `MaybeDeviceZoneScope` below brackets ONE serial stage of the
// reader's per-M-block chain, which is what the "serial composition" question needs measured. They
// are ALWAYS COMPILED (no opt-in define): with the profiler off the macro emits no instructions at
// all, so the shipped kernel is byte-identical to one with no zones — see the durability contract in
// `perf_instrumentation.hpp`. DO NOT DELETE THEM, and give any new fast path its own zone.
//
// ZONE BUDGET: 8 records per M-block on this kernel, against the profiler's 125-per-core cap. So a
// profiled run resolves per-stage time for m_blocks <= 15 (i.e. count <= 3840 at M_BLOCK 8); above
// that the tail is silently dropped and only the whole-kernel duration is meaningful.

// ---------------------------------------------------------------------------
// Compile-time block model. Every trip count and CB increment below is derived
// from these; none is a literal.
// ---------------------------------------------------------------------------
constexpr uint32_t INPUT_FORMAT = get_compile_time_arg_val(0);  // 0 = bf16 RM sticks, 1 = bfp8 tiles
constexpr uint32_t M_T_MAX = get_compile_time_arg_val(1);
constexpr uint32_t LOCAL_EXPERT_ID = get_compile_time_arg_val(2);
constexpr uint32_t EMB_T = get_compile_time_arg_val(3);
constexpr uint32_t HID_T = get_compile_time_arg_val(4);
constexpr uint32_t KR_PAD = get_compile_time_arg_val(5);   // K tiles per row-group slot (uniform)
constexpr uint32_t HN_PAD = get_compile_time_arg_val(6);   // hidden tiles per column-group (uniform)
constexpr uint32_t EC_MAX = get_compile_time_arg_val(7);   // phase-2 N stride (uniform CB increment)
constexpr uint32_t M_BLOCK = get_compile_time_arg_val(8);  // token tile-rows per M-block
constexpr uint32_t HGROUPS = get_compile_time_arg_val(9);
constexpr uint32_t KGROUPS = get_compile_time_arg_val(10);
constexpr uint32_t NUM_BANKS = get_compile_time_arg_val(11);
constexpr uint32_t WRUN = get_compile_time_arg_val(12);  // max bank-contiguous tiles per transaction
constexpr uint32_t SEM_GO = get_compile_time_arg_val(13);
constexpr uint32_t SEM_DATA = get_compile_time_arg_val(14);
// X_PAGE is the ACTIVATION TENSOR's own page (bf16: one full emb stick; bfp8: one tile) — it is
// what TensorAccessor needs to place a page in a bank. X_SLICE is the cb_x_in page stride, i.e.
// only this row-group's KR_PAD-tile slice of a stick. The two are NOT the same number.
constexpr uint32_t X_PAGE = get_compile_time_arg_val(15);
constexpr uint32_t X_SLICE = get_compile_time_arg_val(16);
constexpr uint32_t COUNTS_PAGE = get_compile_time_arg_val(17);
constexpr uint32_t IDX_PAGE = get_compile_time_arg_val(18);
constexpr uint32_t BFP4_TILE = get_compile_time_arg_val(19);
constexpr uint32_t BFP8_TILE = get_compile_time_arg_val(20);
constexpr uint32_t MAX_CHILDREN = get_compile_time_arg_val(21);
constexpr uint32_t REMAP = get_compile_time_arg_val(22);  // 1 = bank-run remap of the N axis
constexpr uint32_t MAILBOX_MAGIC = get_compile_time_arg_val(23);
// W_down prefetch depth in phase-2 K-blocks: how many blocks are kept in flight ahead
// of the round that consumes them. 1 == the per-round read (DRAM-latency bound).
constexpr uint32_t WD_AHEAD = get_compile_time_arg_val(24);
// Smallest legal `m_eff` (= the matmul's out_subblock_h rounded up to a power of two). One
// host-side definition, passed identically to all three kernels — see m_tiles_eff().
constexpr uint32_t M_EFF_MIN = get_compile_time_arg_val(25);
// Concurrent child landing slots in cb_reduce_*_in (Refinement 2 lever 1). A parent invites its
// children in waves of this size instead of one at a time; 1 is the Phase-0 protocol.
constexpr uint32_t REDUCE_SLOTS = get_compile_time_arg_val(26);
// REFINEMENT 3 — CROSS-M-BLOCK WEIGHT RESIDENCY. Every weight read below is a pure function of this
// core's kstart/hstart/jstart with NO M-block index in it, so M-block b > 0 re-reads bytes that are
// still sitting in the same CB slot. With residency on, the reserve/push handshake is kept EXACTLY
// as-is (compute's waits and pops are untouched) and only the DRAM read loops are skipped.
constexpr uint32_t W_RESIDENT = get_compile_time_arg_val(27);   // W_gate (this kernel) + W_up (writer)
constexpr uint32_t WD_RESIDENT = get_compile_time_arg_val(28);  // the phase-2 W_down K stream

constexpr uint32_t cb_x_in = get_compile_time_arg_val(29);
constexpr uint32_t cb_x_tiles = get_compile_time_arg_val(30);
constexpr uint32_t cb_x_stage = get_compile_time_arg_val(31);
constexpr uint32_t cb_w_gate = get_compile_time_arg_val(32);
constexpr uint32_t cb_w_down = get_compile_time_arg_val(33);
constexpr uint32_t cb_reduce_gate_in = get_compile_time_arg_val(34);
constexpr uint32_t cb_reduce_up_in = get_compile_time_arg_val(35);
constexpr uint32_t cb_h = get_compile_time_arg_val(36);
constexpr uint32_t cb_h_local = get_compile_time_arg_val(37);
constexpr uint32_t cb_idx_scratch = get_compile_time_arg_val(38);
constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(39);

// PERF 2 — REDUCE-SCATTER (MOE_SWIGLU_REDUCE=scatter). 1 replaces the tree's parent side with the
// column's peer invite + gather, and makes the root's cb_h_local a pure LANDING region that the
// workers tile directly; 0 reproduces the tree byte-for-byte. See the knob in the program descriptor.
constexpr uint32_t SCATTER = get_compile_time_arg_val(40);
// 1 = the UP half of the gather is issued from HERE (NOC_0) while the writer keeps the GATE half on
// NOC_1 — see MOE_SWIGLU_SCATTER_NOC. Split by payload so each data-movement RISC-V owns ONE CB
// outright (this kernel then pops cb_up_acc and the writer pops cb_gate_acc); two RISC-Vs popping one
// CB corrupts the shared `tiles_acked` word exactly the way two pushers corrupt `tiles_received`.
constexpr uint32_t SCATTER_NOC_SPLIT = get_compile_time_arg_val(41);
constexpr uint32_t GATHER_PAGES = get_compile_time_arg_val(42);  // the WHOLE landing CB, in tiles
constexpr uint32_t SEM_HSLICE = get_compile_time_arg_val(43);
constexpr uint32_t cb_gather_gate = get_compile_time_arg_val(44);
constexpr uint32_t cb_gather_up = get_compile_time_arg_val(45);
constexpr uint32_t cb_up_acc = get_compile_time_arg_val(46);

constexpr uint32_t CT_XMCAST = 47;
constexpr uint32_t CT_HMCAST = CT_XMCAST + 5;

constexpr uint32_t TILE_H = 32;
constexpr uint32_t BF16_TILE_ROW_BYTES = TILE_H * 2;  // one 32-element tile slice of a bf16 stick

// runtime-arg layout
constexpr uint32_t RT_CHILDREN = 16;
// PERF 2 — the whole COLUMN in virtual coordinates, KGROUPS (vx, vy) pairs in ROW order: the invite
// fan-out and (under SCATTER_NOC_SPLIT) the up-gather destinations. Row `r` is at index `r` on every
// core in the column, which is what makes "worker r owns tiles [r*sl_a, (r+1)*sl_a)" agree grid-wide.
constexpr uint32_t RT_PEERS = RT_CHILDREN + 2 * MAX_CHILDREN;
constexpr uint32_t RT_XMCAST = RT_PEERS + 2 * KGROUPS;
constexpr uint32_t RT_HMCAST = RT_XMCAST + 4 + 2 * HGROUPS;

// The x row-multicast: one rotating injector per tile-row over the HGROUPS-wide grid row.
constexpr auto xmc = McastArgs<CT_XMCAST, RT_XMCAST, HGROUPS>();
// The h all-gather: HGROUPS rounds over the whole HGROUPS x KGROUPS grid, round r sent by
// column r's reduce root. SPAN is the rect area (row-major sender list).
constexpr auto hmc = McastArgs<CT_HMCAST, RT_HMCAST, HGROUPS * KGROUPS>();

constexpr uint32_t TA_BASE = CT_HMCAST + 5;
constexpr auto x_args = TensorAccessorArgs<TA_BASE>();
constexpr auto wg_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto wd_args = TensorAccessorArgs<wg_args.next_compile_time_args_offset()>();
constexpr auto cnt_args = TensorAccessorArgs<wd_args.next_compile_time_args_offset()>();
constexpr auto idx_args = TensorAccessorArgs<cnt_args.next_compile_time_args_offset()>();

// Bank-run coalescing (see moe_fused_swiglu_bank_runs.hpp): ONE definition, bound here to this
// kernel's compile-time knobs.
using BR = moe_fused_swiglu::BankRuns<REMAP != 0, NUM_BANKS, WRUN>;

void kernel_main() {
    const uint32_t mailbox_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_gate_addr = get_arg_val<uint32_t>(2);
    const uint32_t w_down_addr = get_arg_val<uint32_t>(3);
    const uint32_t counts_addr = get_arg_val<uint32_t>(4);
    const uint32_t idx_addr = get_arg_val<uint32_t>(5);
    const uint32_t kr = get_arg_val<uint32_t>(6);      // real K tiles this grid ROW owns
    const uint32_t kstart = get_arg_val<uint32_t>(7);  // first emb tile index this row owns
    const uint32_t hstart = get_arg_val<uint32_t>(8);  // first hidden linear index this COLUMN owns
    const uint32_t hn = get_arg_val<uint32_t>(9);      // real hidden tiles this column owns
    const uint32_t ec = get_arg_val<uint32_t>(10);     // output emb tiles this CORE owns
    const uint32_t jstart = get_arg_val<uint32_t>(11);
    const uint32_t is_root = get_arg_val<uint32_t>(12);
    const uint32_t num_children = get_arg_val<uint32_t>(13);
    const uint32_t my_col = get_arg_val<uint32_t>(14);
    // PERF 2 — my row in the grid column: which slice of the reduce-scatter I own (0 tiles = an idle
    // core, which still contributes and still invites).
    const uint32_t my_row = get_arg_val<uint32_t>(15);

    const auto x_acc = TensorAccessor(x_args, x_addr, X_PAGE);
    const auto wg_acc = TensorAccessor(wg_args, w_gate_addr, BFP4_TILE);
    const auto wd_acc = TensorAccessor(wd_args, w_down_addr, BFP4_TILE);
    const auto cnt_acc = TensorAccessor(cnt_args, counts_addr, COUNTS_PAGE);
    const auto idx_acc = TensorAccessor(idx_args, idx_addr, IDX_PAGE);

    // -----------------------------------------------------------------------
    // Phase 0 — the device-resident count. count = counts[ idx[local_expert_id] ].
    // Two one-page reads into unpushed scratch CBs, read back through a volatile L1 pointer.
    // -----------------------------------------------------------------------
    const uint32_t l1_idx = get_write_ptr(cb_idx_scratch);
    noc_async_read(idx_acc.get_noc_addr(0), l1_idx, IDX_PAGE);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t g = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_idx)[LOCAL_EXPERT_ID];

    const uint32_t l1_cnt = get_write_ptr(cb_counts_scratch);
    noc_async_read(cnt_acc.get_noc_addr(0), l1_cnt, COUNTS_PAGE);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_cnt)[g];

    uint32_t m_t = (count + TILE_H - 1) / TILE_H;
    if (m_t > M_T_MAX) {
        m_t = M_T_MAX;
    }
    const uint32_t m_blocks = (m_t + M_BLOCK - 1) / M_BLOCK;

    // Publish {count, M_t, m_blocks} so compute (all three TRISCs) and the writer can read it.
    // The fence between the payload and the READY stamp is the publish barrier: on Blackhole
    // `invalidate_l1_cache()` IS `asm("fence")` (risc_common.h), and L1 is write-through, so the
    // payload words are visible to any other RISC-V that sees MBOX_READY.
    volatile tt_l1_ptr uint32_t* mbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_addr);
    mbox[moe_fused_swiglu::MBOX_COUNT] = count;
    mbox[moe_fused_swiglu::MBOX_M_T] = m_t;
    mbox[moe_fused_swiglu::MBOX_M_BLOCKS] = m_blocks;
    invalidate_l1_cache();
    mbox[moe_fused_swiglu::MBOX_READY] = MAILBOX_MAGIC;

    // -----------------------------------------------------------------------
    // Collective pipes. Receivers are constructed before any ack, so their local flag init is
    // race-free (see mcast_pipe.hpp SEMAPHORE LIFECYCLE).
    // -----------------------------------------------------------------------
    Noc noc;
    auto x_recv = xmc.receiver(noc);
    auto h_recv = hmc.receiver(noc);
    auto x_send = xmc.sender(noc);
    auto h_send = hmc.sender(noc);

    const uint32_t sem_data = static_cast<uint32_t>(get_semaphore(SEM_DATA));
    volatile tt_l1_ptr uint32_t* sem_data_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_data);
    uint32_t data_arrivals = 0;
    // PERF 2 — the h-slice gather counter (scatter path, roots only). Monotone and cumulative.
    volatile tt_l1_ptr uint32_t* sem_h_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_HSLICE)));
    uint32_t h_arrivals = 0;
#if HSEND_WRITER
    // PERF 3 — per-cb_h-slot arrival expectation, this core's own running total. See the wait site.
    uint32_t h_exp[DEPTH_H] = {0};
#endif
#ifdef ABLATE_NO_REDUCE_XFER
    (void)sem_data_ptr;
    (void)data_arrivals;
    (void)sem_h_ptr;
    (void)h_arrivals;
#endif

    constexpr uint32_t SLOTS_H = REMAP ? (HID_T / NUM_BANKS) : HID_T;
    constexpr uint32_t SLOTS_E = REMAP ? (EMB_T / NUM_BANKS) : EMB_T;

    constexpr uint32_t WG_BLOCK_TILES = KR_PAD * HN_PAD;      // one gate weight K-block (num_k_blocks == 1)
    // PERF 3 — the N-chunk the weight stream is published in. GU_CHUNKS == 1 restores the whole-block
    // push byte for byte (the chunk IS the block, and its row stride is HN_PAD again).
    constexpr uint32_t GU_CHUNK_W = HN_PAD / GU_CHUNKS;
    constexpr uint32_t WG_CHUNK_TILES = KR_PAD * GU_CHUNK_W;
    constexpr uint32_t REDUCE_SLOT_TILES = M_BLOCK * HN_PAD;  // one child's landing slot
    constexpr uint32_t REDUCE_CB_TILES = REDUCE_SLOTS * REDUCE_SLOT_TILES;  // the whole CB — see 1c
    constexpr uint32_t X_ROW_BYTES = KR_PAD * BFP8_TILE;

    // `count == 0` -> m_blocks == 0 on every core: no CB traffic, no collective round, no
    // semaphore. Uniform across the grid, so it cannot hang.
    for (uint32_t b = 0; b < m_blocks; ++b) {
        // The RUNTIME token tile-rows this block actually works on. Identical on every core (it is
        // a pure function of the same mailbox words), which is what keeps the three collectives'
        // round counts and landing addresses in lockstep across the grid.
        const uint32_t m_eff = moe_fused_swiglu::m_tiles_eff(m_t, b, M_BLOCK, M_EFF_MIN);
        const uint32_t x_slot_tiles = m_eff * KR_PAD;   // resident in0 block, one slot
        const uint32_t h_block_tiles = m_eff * HN_PAD;  // one phase-2 K-block of h

        // REFINEMENT 3 — the weight DRAM read happens on M-block 0 only when the block is resident.
        // `cb_pop_front` advances a read pointer without touching the bytes, and each weight CB has
        // a single producer, so the slot a later M-block re-reserves still holds what block 0 read
        // into it. Everything else — reserve, push, barrier, trip counts — is unchanged, which is
        // what keeps compute bit-for-bit identical.
        const bool read_wg = (b == 0) || (W_RESIDENT == 0);
        const bool read_wd = (b == 0) || (WD_RESIDENT == 0);

        // -------------------------------------------------------------------
        // Phase 1a — stage x and multicast it along the grid row.
        //
        // cb_x_tiles is ONE slot of M_BLOCK*KR_PAD tiles, so its write pointer is the same L1
        // address on every core in the row (mcast_pipe requires an identical landing address).
        // -------------------------------------------------------------------
        // W_gate is ISSUED HERE, ahead of the staging prologue and the x rounds, and only
        // BARRIERED/pushed after them: with num_k_blocks == 1 the whole block must be fronted
        // before the gate matmul starts, so this is the only place its DRAM latency can overlap
        // anything.
        //
        // REFINEMENT 2 MEASUREMENT, recorded because it is counter-intuitive: the staging
        // prologue's own `noc_async_read_barrier()` calls DRAIN this prefetch (a read barrier is
        // all-or-nothing), so the block is fully paid for before the multicast chain it was
        // designed to hide under even starts. Issuing it AFTER the prologue instead — so the chain
        // really does cover it — was tried and measured WORSE (223 172 / 145 140 / 210 773 ns vs
        // 222 446 / 144 133 / 208 565 on count 256 / 128 / emb 6144): started later, the read no
        // longer overlaps the prologue's OWN stick reads, and that overlap is worth more than the
        // handshake chain's. Kept here. The same defect class in phase 2 — where the barrier had
        // nothing else to cover — is real and is fixed below.
        //
        // PERF 3 — WHERE this issue sits is a KNOB (`XSTAGE_FIRST`), because the barrier below is
        // all-or-nothing and therefore decides which stream the OTHER one waits for. Measured on the
        // focus cell (emb 7168, count 256): with W_gate issued FIRST, `reader_xmcast` finishes at
        // 24.7 us on grid row 11 and 57.7 us on row 2 — a 33 us spread that `compute_gateup` inherits
        // exactly (ends 43 us vs 86 us) — because the staging prologue's `noc_async_read_barrier()`
        // drains this 8.7 MB grid-wide prefetch before a single stick is tilized. x is 3.67 MB and
        // the whole grid needs it before ANY matmul starts; the weights are 16.5 MB and are needed
        // per-column. Issuing the small, universally-blocking stream first is the correct order.
        //
        // PERF 3 — N-CHUNKED WEIGHT STREAM. The block is issued and published in `GU_CHUNKS` slices
        // of the HIDDEN (N) axis, each a full-K [KR_PAD x GU_CHUNK_W] matmul input in its own right,
        // so compute starts on chunk 0 while chunk 1 is still in DRAM. N is the axis that admits this
        // for free: chunks are INDEPENDENT matmuls, so there is no cross-chunk accumulation and no
        // extra L1-accumulating pack — which is exactly why K-chunking was rejected here (it would
        // cost `m_eff` extra accumulating packs per extra K-block, roughly what the overlap wins).
        // Each chunk is CONTIGUOUS in the CB (row stride GU_CHUNK_W, not HN_PAD).
        auto issue_wg_chunk = [&](uint32_t c) {
            cb_reserve_back(cb_w_gate, WG_CHUNK_TILES);
            MaybeDeviceZoneScope("reader_wg_issue");
            const uint32_t wg_wp = get_write_ptr(cb_w_gate);
            const uint32_t h0 = c * GU_CHUNK_W;
            // The ragged last column (hn < HN_PAD) narrows the LAST chunk; the host guarantees every
            // chunk still holds at least one real column, so `w` is never 0.
            uint32_t w = (h0 < hn) ? (hn - h0) : 0;
            if (w > GU_CHUNK_W) {
                w = GU_CHUNK_W;
            }
#ifndef ABLATE_NO_W_XFER  // /perf-measure: drop the weight DRAM stream, keep every CB + barrier
            if (read_wg && w) {
                for (uint32_t k = 0; k < kr; ++k) {
                    BR::read(
                        wg_acc,
                        (kstart + k) * HID_T,
                        hstart + h0,
                        hstart + h0 + w,
                        SLOTS_H,
                        wg_wp + k * GU_CHUNK_W * BFP4_TILE,
                        BFP4_TILE);
                }
            }
#else
            (void)wg_wp;
#endif
        };
#if XSTAGE_FIRST == 0
        issue_wg_chunk(0);
#endif

        cb_reserve_back(cb_x_tiles, x_slot_tiles);
        const uint32_t x_base = get_write_ptr(cb_x_tiles);

        // ---- x staging PROLOGUE: land every tile-row THIS core injects, before the chain ----
        //
        // Staging (the DRAM stick read, the fused tilize, and the landing in the resident slot) is
        // per-injector work with NO cross-core ordering: `dst` is a fixed offset inside the slot
        // reserved just above, and each tile-row has exactly one injector. Hoisting it out of the
        // multicast loop lets all m_eff injectors stage CONCURRENTLY instead of each one stalling
        // its own round while every other core in the row sits in `receive()`. The round loop below
        // is then a pure collective: one multicast per tile-row and nothing else.
        //
        // The landing is a self-copy (bf16) / a direct read (bfp8) rather than a multicast loopback
        // so that the send below is `src == dst` and therefore EXCLUDE-source. That is NOT
        // cosmetic: a `src != dst` send is a LOOPBACK multicast (mcast_pipe.inl
        // `loopback = in_rect_ && src_l1 != dst_l1`), which makes this core's OWN data-ready cell a
        // multicast destination — and the rotating-sender reset right after
        // (`data_ready_.set(INVALID)` behind a `fence_()` that is `async_writes_flushed`, i.e. SENT,
        // not LANDED) then races that in-flight VALID. When the late VALID wins, this core's next
        // `receive()` returns on a stale flag, every later round shifts one early, and the block's
        // LAST tile-row is consumed before its multicast lands: silent, run-to-run garbage.
        // `noc_async_read_barrier()` IS a real arrival guarantee, so landing the data here removes
        // the loopback altogether, for the same NoC bytes (one fewer multicast destination).
        {
            MaybeDeviceZoneScope("reader_xstage");
            for (uint32_t t = my_col; t < m_eff; t += HGROUPS) {
                const uint32_t dst = x_base + t * X_ROW_BYTES;
                uint32_t row = b * M_BLOCK + t;
                if (row >= M_T_MAX) {
                    row = M_T_MAX - 1;  // rows past the sized region are UNDEFINED; stay in bounds
                }
                if constexpr (INPUT_FORMAT == 0) {
                    // bf16 ROW_MAJOR: read this row-group's emb slice of 32 sticks; compute tilizes
                    // them to bfp8 in cb_x_stage; copy the tile-row into the resident slot.
                    cb_reserve_back(cb_x_in, TILE_H);
                    const uint32_t wp = get_write_ptr(cb_x_in);
#ifndef ABLATE_NO_XSTAGE_XFER  // /perf-measure: drop the ACTIVATION DRAM stream, keep the tilize
                    for (uint32_t s = 0; s < TILE_H; ++s) {
                        noc_async_read(
                            x_acc.get_noc_addr(row * TILE_H + s, kstart * BF16_TILE_ROW_BYTES),
                            wp + s * X_SLICE,
                            kr * BF16_TILE_ROW_BYTES);
                    }
#else
                    (void)wp;
#endif
                    noc_async_read_barrier();
                    cb_push_back(cb_x_in, TILE_H);

                    cb_wait_front(cb_x_stage, KR_PAD);
                    noc_async_read(get_noc_addr(get_read_ptr(cb_x_stage)), dst, X_ROW_BYTES);
                    noc_async_read_barrier();
                    cb_pop_front(cb_x_stage, KR_PAD);
                } else {
                    // bfp8_b TILE: the tiles land straight in the resident slot, no tilize.
#ifndef ABLATE_NO_XSTAGE_XFER
                    for (uint32_t i = 0; i < kr; ++i) {
                        noc_async_read(x_acc.get_noc_addr(row * EMB_T + kstart + i), dst + i * BFP8_TILE, BFP8_TILE);
                    }
#endif
                    noc_async_read_barrier();
                }
            }
        }

        // PERF 3 — this core's `x` is off DRAM. Release the writer's W_up stream (XPRIO). A plain
        // volatile store, not a NoC semaphore op: producer and consumer are two RISC-Vs on the SAME
        // core sharing one L1, and this word has exactly one writer. Monotone, so no reset.
#if XPRIO
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_XSTAGED)) = b + 1;
#endif

        // The staging prologue's barriers are behind us, so this prefetch now has the multicast chain
        // AND the whole gate matmul to land under, instead of standing in front of the stick reads.
#if XSTAGE_FIRST
        issue_wg_chunk(0);
#endif

        // ---- x multicast chain ----
        // m_eff rounds, not M_BLOCK: at count 128 (M_t 4) this is HALF the handshake chain and half
        // the staged bytes, and at count 32 an eighth. m_eff divides M_BLOCK, so cb_x_tiles' write
        // pointer stays block-aligned and identical on every core in the row (which mcast_pipe
        // requires of the landing address).
#ifndef ABLATE_NO_X_XFER  // /perf-measure: drop the x transport, keep cb_x_tiles' reserve/push
        {
            MaybeDeviceZoneScope("reader_xmcast");
            if constexpr (xmc.active) {
                for (uint32_t t = 0; t < m_eff; ++t) {
                    const uint32_t round = t % HGROUPS;
                    if (round == my_col) {
                        x_send.send(x_base + t * X_ROW_BYTES, x_base + t * X_ROW_BYTES, X_ROW_BYTES);
                    } else {
                        x_recv.receive(round);
                    }
                }
            }
        }
#endif
        cb_push_back(cb_x_tiles, x_slot_tiles);

        // -------------------------------------------------------------------
        // Phase 1b — W_gate landed under the x rounds; publish it.
        // (W_up is the writer's twin on NoC1 — the dual-NoC split of op_design.md §1.5.)
        // -------------------------------------------------------------------
        // Publish chunk c, then ISSUE chunk c+1 and immediately block on it: the reader sits in DRAM
        // while compute chews chunk c, which is the whole point of the split. Only chunk c is ever
        // outstanding at a barrier, so `noc_async_read_barrier()`'s all-or-nothing drain is exact
        // here rather than the over-wait it is everywhere else in this kernel.
        {
            MaybeDeviceZoneScope("reader_wg_wait");
            for (uint32_t c = 0; c < GU_CHUNKS; ++c) {
                noc_async_read_barrier();
                cb_push_back(cb_w_gate, WG_CHUNK_TILES);
                if (c + 1 < GU_CHUNKS) {
                    issue_wg_chunk(c + 1);
                }
            }
        }

        // -------------------------------------------------------------------
        // Phase 1b' — W_down for ALL WD_AHEAD phase-2 K-blocks, ISSUED as one batch.
        //
        // Read per round (the obvious shape) leaves only HN_PAD transactions of ~1 KB in flight,
        // which is DRAM-LATENCY bound, not bandwidth bound. Issuing WD_AHEAD blocks at once puts
        // WD_AHEAD*HN_PAD transactions in flight and hides the latency behind the reduce-tree
        // handshakes below. WD_AHEAD is a knob: 1 restores the per-round read.
        // -------------------------------------------------------------------
        constexpr uint32_t WD_BLOCK_TILES = HN_PAD * EC_MAX;
        cb_reserve_back(cb_w_down, WD_AHEAD * WD_BLOCK_TILES);
        {
            MaybeDeviceZoneScope("reader_wd_issue");
            const uint32_t wp = get_write_ptr(cb_w_down);
            (void)wp;
            for (uint32_t r = 0; r < WD_AHEAD; ++r) {
                const uint32_t hbase = r * HN_PAD;
                uint32_t hn_r = HN_PAD;
                if (hbase + hn_r > HID_T) {
                    hn_r = HID_T - hbase;
                }
#ifndef ABLATE_NO_W_XFER
                if (read_wd) {
                    for (uint32_t k = 0; k < hn_r; ++k) {
                        // W_down's K axis is `h`'s hidden axis, which the Hn split remapped too, so
                        // the row index goes through the same remap as the N axis.
                        BR::read(
                            wd_acc,
                            BR::remap(hbase + k, SLOTS_H) * EMB_T,
                            jstart,
                            jstart + ec,
                            SLOTS_E,
                            wp + (r * WD_BLOCK_TILES + k * EC_MAX) * BFP4_TILE,
                            BFP4_TILE);
                    }
                }
#endif
            }
        }

        // -------------------------------------------------------------------
        // Phase 1c — the cross-column reduce's RECEIVER side. The invite (SEM_GO) is the flow control
        // that keeps a contributor from overwriting a landing slot compute has not consumed yet, on
        // both paths; the two differ only in WHO invites WHOM (one parent -> its children, versus
        // every core -> its whole column).
        //
        // THE TREE'S PROTOCOL, REFINEMENT 2 LEVER 1 — the invites go out in WAVES of REDUCE_SLOTS,
        // not one at a time.
        // With one slot the parent had to invite child `c`, wait for its ~102 KB, hand it to compute
        // and only then invite `c + 1`: up to 4 SEQUENTIAL round trips per M-block at the root.
        // With REDUCE_SLOTS slots, a wave's children all transfer CONCURRENTLY into disjoint slots
        // (child `c` owns slot `c % REDUCE_SLOTS`, its own runtime arg) and the parent waits once
        // for the whole wave on the monotone SEM_DATA counter.
        //
        // The reserve/push granularity stays the WHOLE CB, which is what preserves the address
        // proxy this transport is built on: the child unicasts to its OWN
        // `get_write_ptr(cb_reduce_*_in)` (+ its slot stride) as a stand-in for the parent's, and
        // that only holds while every push wraps the write pointer back to the CB base — which a
        // whole-CB push does on every core, whatever its child count. It is also why the m_eff
        // shrink does not move THIS accounting: a slot stays M_BLOCK*HN_PAD tiles and the child
        // ships only the m_eff*HN_PAD tiles that carry live tokens (compute drops the tail).
        // -------------------------------------------------------------------
        // PERF 2 — MY SLICE of this block, from the ONE shared plan (moe_fused_swiglu_common.hpp),
        // identical on every core and every RISC-V. `slice_workers` cores own `slice_tiles` each.
        const uint32_t sl_w = (SCATTER != 0) ? moe_fused_swiglu::slice_workers(h_block_tiles, KGROUPS) : 0;
        const uint32_t sl_a = (sl_w != 0) ? (h_block_tiles / sl_w) : 0;  // uniform slice size
        const uint32_t slice_tiles = (my_row < sl_w) ? sl_a : 0;         // 0 = an idle core
        if constexpr (SCATTER) {
            MaybeDeviceZoneScope("reader_reduce");
            // RECEIVER side of the reduce-scatter. Reserve the landing CBs WHOLE first (so every
            // contributor's own-write-pointer proxy is the CB base on every core, at every m_eff),
            // then invite the WHOLE COLUMN — the generalisation of the tree's parent-invites-child
            // SEM_GO, and the flow control that keeps M-block b+1's contribution from overwriting a
            // slot compute has not consumed. Then wait for every contributor and push WHOLE.
            //
            // This is ALSO the flow control for cb_h_local: my invite for block b+1 is issued here,
            // strictly after my phase 2 of block b has read (and barriered) cb_h_local, and no
            // worker's h-slice send for b+1 can precede this invite (its epilogue is downstream of
            // the gather, which is downstream of every core's invite). So the h landing needs no
            // second handshake.
            if (slice_tiles) {
                cb_reserve_back(cb_gather_gate, GATHER_PAGES);
                cb_reserve_back(cb_gather_up, GATHER_PAGES);
            }
#ifndef ABLATE_NO_REDUCE_XFER  // /perf-measure: drop the all-to-all, keep every CB cycle
            const uint32_t sem_go = static_cast<uint32_t>(get_semaphore(SEM_GO));
            for (uint32_t p = 0; p < KGROUPS; ++p) {
                const uint32_t px = get_arg_val<uint32_t>(RT_PEERS + 2 * p + 0);
                const uint32_t py = get_arg_val<uint32_t>(RT_PEERS + 2 * p + 1);
                noc_semaphore_inc(get_noc_addr(px, py, sem_go), 1);
            }
            noc_async_atomic_barrier();
#endif
            // The UP half of the gather, when MOE_SWIGLU_SCATTER_NOC=split puts it on NOC_0. The
            // GATE half stays on the writer's NOC_1; each RISC-V owns one accumulator CB outright.
            if constexpr (SCATTER_NOC_SPLIT) {
                const uint32_t up_bytes = sl_a * BFP8_TILE;
                cb_wait_front(cb_up_acc, h_block_tiles);
#ifndef ABLATE_NO_REDUCE_XFER
                // Wait for the WHOLE column's invites before writing, exactly as the writer does:
                // every core invites once per peer per M-block, so (b+1)*KGROUPS is the exact total.
                volatile tt_l1_ptr uint32_t* go_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_GO)));
                noc_semaphore_wait_min(go_ptr, (b + 1) * KGROUPS);
                const uint32_t usrc = get_read_ptr(cb_up_acc);
                const uint32_t udst = get_write_ptr(cb_gather_up);
                const uint32_t slot_bytes = my_row * up_bytes;
                const uint32_t sem_data_id = static_cast<uint32_t>(get_semaphore(SEM_DATA));
                for (uint32_t w = 0; w < sl_w; ++w) {
                    const uint32_t vx = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 0);
                    const uint32_t vy = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 1);
                    noc_async_write(usrc + w * up_bytes, get_noc_addr(vx, vy, udst + slot_bytes), up_bytes);
                }
                noc_async_write_barrier();
                for (uint32_t w = 0; w < sl_w; ++w) {
                    const uint32_t vx = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 0);
                    const uint32_t vy = get_arg_val<uint32_t>(RT_PEERS + 2 * w + 1);
                    noc_semaphore_inc(get_noc_addr(vx, vy, sem_data_id), 1);
                }
                noc_async_atomic_barrier();
#endif
                cb_pop_front(cb_up_acc, h_block_tiles);
            }
            if (slice_tiles) {
#ifndef ABLATE_NO_REDUCE_XFER
                // One signal per contributor per payload: KGROUPS under the single-NoC shape, 2 x
                // KGROUPS when the up half is split off, so the total is what says "everything landed".
                data_arrivals += KGROUPS * (SCATTER_NOC_SPLIT ? 2 : 1);
                noc_semaphore_wait_min(sem_data_ptr, data_arrivals);
#endif
                cb_push_back(cb_gather_gate, GATHER_PAGES);
                cb_push_back(cb_gather_up, GATHER_PAGES);
            }
        } else {
            MaybeDeviceZoneScope("reader_reduce");
            // THE TREE, PARENT SIDE — the invite waves described above.
            for (uint32_t c0 = 0; c0 < num_children; c0 += REDUCE_SLOTS) {
                uint32_t wave = num_children - c0;
                if (wave > REDUCE_SLOTS) {
                    wave = REDUCE_SLOTS;
                }
                cb_reserve_back(cb_reduce_gate_in, REDUCE_CB_TILES);
                cb_reserve_back(cb_reduce_up_in, REDUCE_CB_TILES);
#ifndef ABLATE_NO_REDUCE_XFER  // /perf-measure: drop invite + data wait, keep the CB cycle
                for (uint32_t c = c0; c < c0 + wave; ++c) {
                    const uint32_t cx = get_arg_val<uint32_t>(RT_CHILDREN + 2 * c + 0);
                    const uint32_t cy = get_arg_val<uint32_t>(RT_CHILDREN + 2 * c + 1);
                    noc_semaphore_inc(get_noc_addr(cx, cy, static_cast<uint32_t>(get_semaphore(SEM_GO))), 1);
                }
                data_arrivals += wave;
                noc_semaphore_wait_min(sem_data_ptr, data_arrivals);
#endif
                cb_push_back(cb_reduce_gate_in, REDUCE_CB_TILES);
                cb_push_back(cb_reduce_up_in, REDUCE_CB_TILES);
            }
        }

        // -------------------------------------------------------------------
        // Phase 2 — round r of the h all-gather, with W_down already in flight. The gather rides
        // the phase-2 K stream, so it overlaps `down` compute and flow-controls itself on cb_h.
        // The batch issued above landed under the reduce handshakes; publish it, then stream the
        // remaining K-blocks one round ahead of the round that consumes them.
        //
        // REFINEMENT 2 — DEFERRED READ BARRIER. The round's W_down block used to be issued AND
        // barriered inside the same round, before the collective: `noc_async_read_barrier()` drains
        // EVERY outstanding read, so the block's DRAM latency was paid on the spot with nothing else
        // in flight, on all 110 cores, once per round. That is why `WD_AHEAD` measured neutral at
        // Phase 0 — a deeper prefetch cannot help while the barrier that drains it sits one
        // instruction later. Now the issue moves AFTER the round's send/receive and its barrier
        // moves to the NEXT round, so the read lands underneath a whole grid-wide multicast.
        // `wd_pending` carries the one block that is issued-but-not-yet-published across the round
        // boundary; the last issue is at round HGROUPS-1-WD_AHEAD, so it is always published inside
        // the loop and nothing leaks out.
        // -------------------------------------------------------------------
        {
            MaybeDeviceZoneScope("reader_wd_wait");
            noc_async_read_barrier();
        }
        cb_push_back(cb_w_down, WD_AHEAD * WD_BLOCK_TILES);
        // PERF 2 — on the scatter path this column's h block is ASSEMBLED IN cb_h_local BY THE
        // WORKERS' NoC WRITES, not packed by compute, so the root's handshake is the SEM_HSLICE
        // counter rather than a CB front. `sl_w` slices land per M-block; the counter is monotone and
        // cumulative, like every other semaphore in this op.
        if (SCATTER && is_root) {
            h_arrivals += sl_w;
        }
        {
            MaybeDeviceZoneScope("reader_phase2");
            bool wd_pending = false;
            for (uint32_t r = 0; r < HGROUPS; ++r) {
                // This round's cb_h slot is reserved first so the round's sender can ISSUE its self-copy
                // before the barrier below, which then covers the self-copy AND the previous round's
                // W_down block in one drain.
                cb_reserve_back(cb_h, h_block_tiles);
                const uint32_t hdst = get_write_ptr(cb_h);
#if HSEND_WRITER
                // PERF 3 — DUAL-RISC SPLIT, receive side. The send lives on the writer now, so this
                // loop only receives. Two steps, and the ORDER of them is the whole protocol:
                //
                //   1. ACK FIRST. `cb_reserve_back` above is the proof that THIS core's slot
                //      `r % DEPTH_H` is free, so tell round r's sender it may write. Acking before
                //      waiting is what makes the split deadlock-free: the sender needs this core's
                //      ack, and this core must not be blocked on the sender's data to give it.
                //   2. THEN wait on the slot's own arrival counter. Per-slot, because senders are no
                //      longer serialised by the loop and their increments can interleave.
                //
                // Round r's sender is column r's reduce root, core (r, r % KGROUPS); the h mcast's
                // rotating-sender coord table is row-major over the grid and already in RT.
                {
                    const uint32_t sidx = (r % KGROUPS) * HGROUPS + r;
                    const uint32_t svx = get_arg_val<uint32_t>(RT_HMCAST + 4 + 2 * sidx + 0);
                    const uint32_t svy = get_arg_val<uint32_t>(RT_HMCAST + 4 + 2 * sidx + 1);
                    noc_semaphore_inc(get_noc_addr(svx, svy, static_cast<uint32_t>(get_semaphore(SEM_H_FREE))), 1);
                }
#endif
                const bool i_send = (is_root && r == my_col);
                if (i_send) {
                    // Self-copy cb_h_local -> this round's cb_h slot, so the send below is `src == dst`
                    // and therefore EXCLUDE-source. Identical reasoning to the x send above: a
                    // `src != dst` send is a LOOPBACK multicast, and mcast_pipe's rotating-sender flag
                    // reset then races this core's own in-flight VALID, so the root's remaining
                    // `receive()` calls shift one round early and the LAST h K-block is consumed before
                    // it lands. Measured: with the x send fixed but this one left as a loopback, PCC on
                    // a fixed input still varied 0.959-0.979 run to run on BOTH activation formats (the
                    // bfp8 path has no x loopback, so this send was the only remaining suspect); with
                    // both fixed it is bit-stable. Same NoC bytes, one fewer mcast destination.
                    if constexpr (SCATTER) {
                        // The workers' finished slices already TILE cb_h_local (each wrote
                        // `[row*sl_a, (row+1)*sl_a)` of the block), so nothing on the compute thread
                        // produces it: no push, no pop, no CB front. That is exactly why the write
                        // pointer is the CB base on every core every M-block, which is what let the
                        // workers use their OWN pointer as this core's landing address.
#ifndef ABLATE_NO_REDUCE_XFER  // the workers' sends are stubbed too, so this wait must go with them
                        noc_semaphore_wait_min(sem_h_ptr, h_arrivals);
#endif
                        noc_async_read(get_noc_addr(get_write_ptr(cb_h_local)), hdst, h_block_tiles * BFP8_TILE);
                    } else {
                        cb_wait_front(cb_h_local, h_block_tiles);
                        noc_async_read(get_noc_addr(get_read_ptr(cb_h_local)), hdst, h_block_tiles * BFP8_TILE);
                    }
                }

                if (i_send) {
                    // The SENDER must drain before it broadcasts (its self-copy has to have landed), so
                    // it also publishes the pending W_down block here. One core per round pays this.
                    noc_async_read_barrier();  // the self-copy AND the previous round's W_down block
                    if (wd_pending) {
                        cb_push_back(cb_w_down, WD_BLOCK_TILES);
                        wd_pending = false;
                    }
                    if constexpr (SCATTER == 0) {
                        cb_pop_front(cb_h_local, h_block_tiles);
                    }
#ifndef ABLATE_NO_H_XFER  // /perf-measure: drop the h transport, keep cb_h's reserve/push
#if !HSEND_WRITER
                    if constexpr (hmc.active) {
                        h_send.send(hdst, hdst, h_block_tiles * BFP8_TILE);
                    }
#else
                    // PERF 3 — the broadcast to the other 109 cores is the WRITER's job now; this
                    // core only lands its own copy (the self-copy above). Nothing to do here.
                    (void)hdst;
#endif
#endif
                } else {
                    // The other 109 cores drain AFTER the multicast, so the previous round's W_down read
                    // had this whole round's grid-wide broadcast to land under — that is the deferral.
#ifndef ABLATE_NO_H_XFER
#if !HSEND_WRITER
                    if constexpr (hmc.active) {
                        // Round r's sender is column r's root, core (r, r % KGROUPS); the rotating
                        // sender list is row-major over the rect.
                        h_recv.receive((r % KGROUPS) * HGROUPS + r);
                    }
#else
                    // PERF 3 — wait on THIS SLOT's own monotone arrival counter. `h_exp[]` is this
                    // core's private expectation: it is bumped only for rounds this core actually
                    // waits on, which is what keeps a ROOT (which self-copies its own round and is
                    // excluded from its own multicast, so its counter for that slot is one behind
                    // everyone else's) consistent with the rest of the grid without a special case.
                    {
                        const uint32_t s = r % DEPTH_H;
                        noc_semaphore_wait_min(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                static_cast<uint32_t>(get_semaphore(SEM_H_RDY_BASE + s))),
                            ++h_exp[s]);
                    }
#endif
#endif
                    if (wd_pending) {
                        noc_async_read_barrier();
                        cb_push_back(cb_w_down, WD_BLOCK_TILES);
                        wd_pending = false;
                    }
                }

                // ISSUE the next K-block only now — after this round's collective — so its latency is
                // paid under the NEXT round's, not under this one's barrier. Stays WD_AHEAD K-blocks
                // ahead of the round being consumed; WD_AHEAD >= 2 is what actually decouples compute
                // from this block (at 1 the block published here is the one this round consumes).
                const uint32_t pre = r + WD_AHEAD;
                if (pre < HGROUPS) {
                    const uint32_t hbase = pre * HN_PAD;
                    uint32_t hn_r = HN_PAD;
                    if (hbase + hn_r > HID_T) {
                        hn_r = HID_T - hbase;
                    }
                    cb_reserve_back(cb_w_down, WD_BLOCK_TILES);
                    const uint32_t wp = get_write_ptr(cb_w_down);
#ifndef ABLATE_NO_W_XFER
                    if (read_wd) {
                        for (uint32_t k = 0; k < hn_r; ++k) {
                            BR::read(
                                wd_acc,
                                BR::remap(hbase + k, SLOTS_H) * EMB_T,
                                jstart,
                                jstart + ec,
                                SLOTS_E,
                                wp + k * EC_MAX * BFP4_TILE,
                                BFP4_TILE);
                        }
                    }
#else
                    (void)wp;
#endif
                    wd_pending = true;
                }
                cb_push_back(cb_h, h_block_tiles);
            }
            // The last issue is at round HGROUPS-1-WD_AHEAD and is published at round
            // HGROUPS-WD_AHEAD <= HGROUPS-1, so nothing is ever left pending here. Kept as an
            // ASSERT-free safety drain for a future WD_AHEAD that changes that arithmetic.
            if (wd_pending) {
                noc_async_read_barrier();
                cb_push_back(cb_w_down, WD_BLOCK_TILES);
            }
        }
    }
}
