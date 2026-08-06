// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — READER (NoC0).
//
// Per M-block, in order:
//   0. the device-resident token count (idx -> counts[idx[local_expert_id]]), published to the L1
//      mailbox every other kernel spins on;
//   1a. stage this core's `x` tile-rows from DRAM, then ROW-multicast them along grid row y
//      (rotating injector, one tile-row per round) via mcast_pipe. W_gate is issued here too;
//   1b. publish the W_gate stream chunk by chunk (W_up is the writer's twin on NoC1);
//   1b'. issue the phase-2 W_down batch, so it lands under the reduce rendezvous;
//   1c. the reduce-scatter's RECEIVER side: reserve the landing CBs whole, invite the whole
//      column, wait for every contributor, push whole. Also sends the UP half of the gather;
//   2. the phase-2 loop: one W_down K-block plus one round of the grid-wide `h` all-gather per
//      iteration, with a posted per-slot-flag multicast.
//
// RAW-DATAFLOW DEVIATIONS, each because no in-tree helper expresses the thing:
//   * weight/activation/output DRAM traffic uses raw `noc_async_read` over a contiguous RUN of
//     pages; the page-granular TensorAccessor helper issues one transaction per tile, which is
//     transaction-rate-bound at ~5 GB/s/core. Address computation still goes through TensorAccessor.
//   * the reduce-scatter transport is raw unicast + counting semaphores: mcast_pipe's SenderPipe is
//     a rectangle multicast, while a gather leg is point-to-point with a different destination per
//     peer, and the fan-in needs counting.
//   * the h multicast is raw because `noc.h` blocks POSTED multicast at the library level.
//   * the token-count publish is a raw L1 mailbox because compute needs a scalar loop bound on ALL
//     THREE TRISCs and `cb_wait_front` in a compute kernel is UNPACK-only.
//
// The measurement behind every choice here is in perf_experiments/DESIGN_NOTES.md.
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

#include "moe_fused_swiglu_dataflow.hpp"  // the transport vocabulary shared with the writer
#include "moe_fused_swiglu_common.hpp"     // the ONE definition of the mailbox word layout
#include "moe_fused_swiglu_ct_args.hpp"    // the ONE definition of the compile-time arg order

using namespace dataflow_kernel_lib;

// PER-STAGE ZONES — PERMANENT, always compiled. With the profiler off the macro emits no
// instructions, so the shipped kernel is byte-identical to one with no zones. DO NOT DELETE THEM,
// and give any new fast path its own zone. Budget: 8 records per M-block against a 125-per-core
// cap, so a profiled run resolves per-stage time for m_blocks <= 15.

// ---------------------------------------------------------------------------
// Compile-time block model. Every trip count and CB increment below is derived
// from these; none is a literal.
// ---------------------------------------------------------------------------
MOE_DECLARE_CT_ENUM(MOE_READER_CT_ARGS);

constexpr uint32_t INPUT_FORMAT = CT(INPUT_FORMAT);  // 0 = bf16 RM sticks, 1 = bfp8 tiles
constexpr uint32_t M_T_MAX = CT(M_T_MAX);
constexpr uint32_t LOCAL_EXPERT_ID = CT(LOCAL_EXPERT_ID);
constexpr uint32_t EMB_T = CT(EMB_T);
constexpr uint32_t HID_T = CT(HID_T);
constexpr uint32_t KR_PAD = CT(KR_PAD);  // K tiles per row-group slot (uniform)
constexpr uint32_t HN_PAD = CT(HN_PAD);  // hidden tiles per column-group (uniform)
constexpr uint32_t EC_MAX = CT(EC_MAX);  // phase-2 N stride (uniform CB increment)
constexpr uint32_t M_BLOCK = CT(M_BLOCK);
constexpr uint32_t HGROUPS = CT(HGROUPS);
constexpr uint32_t KGROUPS = CT(KGROUPS);
constexpr uint32_t NUM_CORES = CT(NUM_CORES);

constexpr uint32_t SEM_GO = CT(SEM_GO);
constexpr uint32_t SEM_DATA = CT(SEM_DATA);
constexpr uint32_t SEM_HSLICE = CT(SEM_HSLICE);
constexpr uint32_t SEM_XSTAGED = CT(SEM_XSTAGED);
constexpr uint32_t SEM_H_RDY_BASE = CT(SEM_H_RDY_BASE);
constexpr uint32_t SEM_H_FREE = CT(SEM_H_FREE);
constexpr uint32_t SEM_WDSPLIT = CT(SEM_WDSPLIT);
constexpr uint32_t SEM_HROW_FREE = CT(SEM_HROW_FREE);
constexpr uint32_t SEM_PHASE_FREE = CT(SEM_PHASE_FREE);
constexpr uint32_t PHASE_CB_ALIAS = CT(PHASE_CB_ALIAS);

// X_PAGE is the ACTIVATION TENSOR's own page (bf16: one full emb stick; bfp8: one tile) — what
// TensorAccessor needs to place a page in a bank. X_SLICE is the cb_x_in page stride, i.e. only
// this row-group's KR_PAD-tile slice of a stick. Not the same number.
constexpr uint32_t X_PAGE = CT(X_PAGE);
constexpr uint32_t X_SLICE = CT(X_SLICE);
constexpr uint32_t COUNTS_PAGE = CT(COUNTS_PAGE);
constexpr uint32_t IDX_PAGE = CT(IDX_PAGE);
constexpr uint32_t W_TILE = CT(W_TILE_BYTES);  // weight tile stride: bfp4 576, bfp8 1088, bf16 2048
constexpr uint32_t BFP8_TILE = CT(BFP8_TILE);
// h is bfp8, like x, the output and the reduce operands. bfp4 h was measured and is a precision
// failure (pcc=nan on 8 of 9 cells: the packer emits bfp8 into a bfp4 CB).
constexpr uint32_t H_TILE = BFP8_TILE;
constexpr uint32_t MAILBOX_MAGIC = CT(MAILBOX_MAGIC);

// W_down blocks kept in flight ahead of the round that consumes them; 1 == the per-round read.
constexpr uint32_t WD_AHEAD = CT(WD_AHEAD);
// Smallest legal `m_eff`. One host definition, identical in all three kernels — see m_tiles_eff().
constexpr uint32_t M_EFF_MIN = CT(M_EFF_MIN);
// Cross-M-block weight residency: every weight read is a pure function of this core's
// kstart/hstart/jstart with no M-block index, so b > 0 re-reads bytes still in the CB slot. The
// reserve/push handshake is untouched; only the DRAM read loops are skipped.
constexpr uint32_t W_RESIDENT = CT(W_RESIDENT);
constexpr uint32_t WD_RESIDENT = CT(WD_RESIDENT);
constexpr uint32_t WD_MROW_ROUNDS = CT(WD_MROW_ROUNDS);
constexpr uint32_t GU_CHUNKS = CT(GU_CHUNKS);
constexpr uint32_t XPRIO = CT(XPRIO);
constexpr uint32_t HACK_AHEAD = CT(HACK_AHEAD);
constexpr uint32_t DEPTH_H = CT(DEPTH_H);
constexpr uint32_t WD_SPLIT = CT(WD_SPLIT);
constexpr uint32_t WG_SHARD_W = CT(WG_SHARD_W);
constexpr uint32_t WD_SHARD_W = CT(WD_SHARD_W);
constexpr uint32_t GATHER_PAGES = CT(GATHER_PAGES);  // the WHOLE landing CB, in tiles

constexpr uint32_t cb_x_in = CT(CB_X_IN);
constexpr uint32_t cb_x_tiles = CT(CB_X_TILES);
constexpr uint32_t cb_x_ready = CT(CB_X_STAGE);  // legacy host/CT name; carries completion, not x payload
constexpr uint32_t cb_w_gate = CT(CB_W_GATE);
constexpr uint32_t cb_w_down = CT(CB_W_DOWN);
constexpr uint32_t cb_h = CT(CB_H);
constexpr uint32_t cb_h_local = CT(CB_H_LOCAL);
constexpr uint32_t cb_idx_scratch = CT(CB_IDX_SCRATCH);
constexpr uint32_t cb_counts_scratch = CT(CB_COUNTS_SCRATCH);
constexpr uint32_t cb_gather_gate = CT(CB_GATHER_GATE);
constexpr uint32_t cb_gather_up = CT(CB_GATHER_UP);
constexpr uint32_t cb_up_acc = CT(CB_UP_ACC);

// The mcast and TensorAccessor blocks follow the scalar block; `CT_COUNT` is its length, so
// inserting a scalar argument no longer shifts them.
constexpr uint32_t CT_XMCAST = CT_COUNT;
constexpr uint32_t CT_HMCAST = CT_XMCAST + 5;

constexpr uint32_t TILE_H = 32;
constexpr uint32_t BF16_TILE_ROW_BYTES = TILE_H * 2;  // one 32-element tile slice of a bf16 stick

// Cross-block x prefetch uses the two highest read transaction IDs. Phase-2 reads must be tagged
// separately because a blanket read barrier drains EVERY id and would turn the prefetch back into
// a serial read. The IDs are local to this data-movement RISC-V.
constexpr bool kPrefetchNextX = true;
constexpr uint32_t P2_READ_TRID = 14;
constexpr uint32_t NEXT_X_TRID = 15;

// Runtime-arg layout. The scalar block, then the whole COLUMN in virtual coordinates as KGROUPS
// (vx, vy) pairs in ROW order: the invite fan-out and the up-gather destinations. Row `r` is at
// index `r` on every core in the column, which is what makes "worker r owns tiles
// [r*sl_a, (r+1)*sl_a)" agree grid-wide.
constexpr uint32_t RT_PEERS = 14;
constexpr uint32_t RT_XMCAST = RT_PEERS + 2 * KGROUPS;
constexpr uint32_t RT_HMCAST = RT_XMCAST + 4 + 2 * HGROUPS;

// The x row-multicast: one rotating injector per tile-row over the HGROUPS-wide grid row.
constexpr auto xmc = McastArgs<CT_XMCAST, RT_XMCAST, HGROUPS>();
// The h all-gather: HGROUPS rounds over the whole HGROUPS x KGROUPS grid, round r sent by
// column r's reduce root. SPAN is the rect area (row-major sender list).
constexpr auto hmc = McastArgs<CT_HMCAST, RT_HMCAST, HGROUPS * KGROUPS>();

// The h multicast. `H_MCAST_POSTED` selects the ONE wire property that differs between the two
// variants — whether the payload write posts — and nothing else: the linked chain, the flag, the
// exclude-source fan-out and the barrier structure below are shared, so an A/B on this define is
// an A/B on posting alone.
//
// POSTED (1, default). The Flag path already takes no acked write barrier, but the write is
// non-posted, so all NUM_CORES-1 destinations return write-acks. Posting removes that traffic and
// nothing else: the VALID flag stays NON-posted and LINKED on the same VC, so it cannot overtake
// the payload — which is the invariant mcast_pipe's Flag path already depends on today. Posting
// changes whether acks come back, never the wire order. See DESIGN_NOTES.md §3.
//
// NON-POSTED (0). The conservative fallback, and what mcast_pipe's library path already does for
// the x multicast: acks return and the payload write is tracked, so correctness no longer rests
// on the linked-chain ordering guarantee alone. Keep this reachable — the posted variant's failure
// mode would be a receiver reading a half-written slot, i.e. a SILENT PCC drift rather than a
// hang, so the cheap way to test the ordering claim is to measure both.
constexpr bool kHMcastPosted = (H_MCAST_POSTED != 0);

inline void h_slot_send_posted(uint32_t slot, uint32_t l1, uint32_t size) {
    const auto hrect = hmc.template rect<noc_index>();
    const auto& rb = hrect.bounds();
    // EXCLUDE-source: `src == dst` on this send (the self-copy already placed this core's own copy),
    // so the fan-out is the rect area minus this core — the same count SenderPipe's
    // `num_dests_excl_` computes, and the same reason it is not a loopback (§4 trap 4).
    const uint32_t ndest = hrect.area() - 1;
    const uint32_t hf_addr = static_cast<uint32_t>(get_semaphore(SEM_H_RDY_BASE + slot));
    volatile tt_l1_ptr uint32_t* hf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hf_addr);

    // 1. the payload — LINKED, so the flag below cannot overtake it. POSTED (no return acks) only
    //    when kHMcastPosted; otherwise all `ndest` destinations ack, which is the conservative
    //    variant. Identical in every other argument.
    ncrisc_noc_fast_write_any_len<noc_mode>(
        noc_index,
        write_cmd_buf,
        l1,
        get_noc_multicast_addr(rb.sx, rb.sy, rb.ex, rb.ey, l1),
        size,
        NOC_MULTICAST_WRITE_VC,
        /*mcast=*/true,
        /*linked=*/true,
        ndest,
        /*multicast_path_reserve=*/true,
        /*posted=*/kHMcastPosted);
    // 2. re-assert VALID locally: `set_multicast` broadcasts THIS core's own cell as the source, and
    //    a core that also receives on this cell left it INVALID after its last receive.
    noc_semaphore_set(hf, VALID);
    // 3. the flag — NON-POSTED, on the same VC, terminating the link. This is the arrival proof.
    noc_semaphore_set_multicast(
        hf_addr, get_noc_multicast_addr(rb.sx, rb.sy, rb.ex, rb.ey, hf_addr), ndest, /*linked=*/false);
    // 4. SENT, so the flag cell is safe to rewrite; 5. rotating-sender reset.
    noc_async_writes_flushed();
    noc_semaphore_set(hf, INVALID);
}

constexpr uint32_t TA_BASE = CT_HMCAST + 5;
constexpr auto x_args = TensorAccessorArgs<TA_BASE>();
constexpr auto wg_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto wd_args = TensorAccessorArgs<wg_args.next_compile_time_args_offset()>();
constexpr auto cnt_args = TensorAccessorArgs<wd_args.next_compile_time_args_offset()>();
constexpr auto idx_args = TensorAccessorArgs<cnt_args.next_compile_time_args_offset()>();

// Three `WeightRuns` bindings, because the three tensors this kernel touches can have DIFFERENT
// placements: each weight stream takes its own tensor's DRAM ND shard width (0 = interleaved,
// one transaction per tile), and everything else stays on the interleaved binding.
using BR = moe_fused_swiglu::WeightRuns<>;
using BRG = moe_fused_swiglu::WeightRuns<WG_SHARD_W>;  // W_gate
using BRD = moe_fused_swiglu::WeightRuns<WD_SHARD_W>;  // W_down

// The W_down NoC split: WD_SPLIT eighths of every phase-2 K-block's hidden rows are read by the
// WRITER on NOC_1; this kernel keeps the head rows. The writer takes the TAIL rows so both sides
// read a contiguous run and the coalescing is unchanged on either side.
inline uint32_t wd_rows_writer(uint32_t hn_r) { return (hn_r * WD_SPLIT) / 8; }

// The cross-RISC completion gate. `noc_async_read_barrier()` is PER-RISC-V, so this kernel's
// barrier proves nothing about the writer's share of the SAME K-blocks; publishing without this
// hands the `down` matmul a half-written weight tile. `wd_done` is this kernel's running total of
// published blocks and the counter is the writer's of completed ones — both monotone, so waiting
// for `wd_done + n` before pushing n blocks is the tightest legal gate.
inline void wd_split_gate(uint32_t& wd_done, uint32_t n) {
    wd_done += n;
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_WDSPLIT)), wd_done);
}

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
    const uint32_t my_col = get_arg_val<uint32_t>(12);
    // My row in the grid column: which slice of the reduce-scatter I own (0 tiles = an idle core,
    // which still contributes and still invites).
    const uint32_t my_row = get_arg_val<uint32_t>(13);
    // Column `x`'s reduce root is row `x % KGROUPS` — the core that injects this column's h into
    // the phase-2 all-gather. Derived, not passed: one rule, three kernels.
    const bool is_root = (my_row == my_col % KGROUPS);
    const bool is_row_agg = (my_col == my_row);

    const auto x_acc = TensorAccessor(x_args, x_addr, X_PAGE);
    const auto wg_acc = TensorAccessor(wg_args, w_gate_addr, W_TILE);
    const auto wd_acc = TensorAccessor(wd_args, w_down_addr, W_TILE);
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
    moe_fused_swiglu::mailbox_publish(mailbox_addr, MAILBOX_MAGIC, count, m_t, m_blocks);

    // -----------------------------------------------------------------------
    // Collective pipes. Receivers are constructed before any ack, so their local flag init is
    // race-free (see mcast_pipe.hpp SEMAPHORE LIFECYCLE).
    // -----------------------------------------------------------------------
    Noc noc;
    auto x_recv = xmc.receiver(noc);
    auto x_send = xmc.sender(noc);

    const uint32_t sem_data = static_cast<uint32_t>(get_semaphore(SEM_DATA));
    volatile tt_l1_ptr uint32_t* sem_data_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_data);
    uint32_t data_arrivals = 0;
    // PERF 2 — the h-slice gather counter (scatter path, roots only). Monotone and cumulative.
    volatile tt_l1_ptr uint32_t* sem_h_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_HSLICE)));
    uint32_t h_arrivals = 0;
    // Monotone ack accounting for the per-slot-flag h send. Each core acks each round's root
    // exactly once per M-block, so a root's cell gains exactly NUM_CORES per M-block whatever
    // order the acks arrive in. SEM_H_FREE is monotone ACROSS M-blocks, so this must be too.
    uint32_t h_free_expected = 0;
    // W_down K-blocks this kernel has already PUBLISHED, running across M-blocks so it stays
    // aligned with the writer's equally-monotone completion counter.
    uint32_t wd_done = 0;
#ifdef ABLATE_NO_REDUCE_XFER
    (void)sem_data_ptr;
    (void)data_arrivals;
    (void)sem_h_ptr;
    (void)h_arrivals;
#endif

    constexpr uint32_t WG_BLOCK_TILES = KR_PAD * HN_PAD;      // one gate weight K-block (num_k_blocks == 1)
    // PERF 3 — the N-chunk the weight stream is published in. GU_CHUNKS == 1 restores the whole-block
    // push byte for byte (the chunk IS the block, and its row stride is HN_PAD again).
    constexpr uint32_t GU_CHUNK_W = HN_PAD / GU_CHUNKS;
    constexpr uint32_t WG_CHUNK_TILES = KR_PAD * GU_CHUNK_W;
    constexpr uint32_t REDUCE_SLOT_TILES = M_BLOCK * HN_PAD;  // one child's landing slot
    constexpr uint32_t X_ROW_BYTES = KR_PAD * BFP8_TILE;

    // One activation-row issue body for both the ordinary prologue and the cross-block prefetch.
    // On the bf16 path `dst` is one cb_x_in stick-row slot; on the tiled path it is the resident
    // cb_x_tiles row. Completion and CB publication stay with the caller.
    auto issue_x_row = [&](uint32_t row, uint32_t dst) {
#ifndef ABLATE_NO_XSTAGE_XFER
        if constexpr (INPUT_FORMAT == 0) {
            for (uint32_t i = 0; i < TILE_H; ++i) {
                const uint32_t s = (i + my_col + my_row) % TILE_H;
                noc_async_read(
                    x_acc.get_noc_addr(row * TILE_H + s, kstart * BF16_TILE_ROW_BYTES),
                    dst + s * X_SLICE,
                    kr * BF16_TILE_ROW_BYTES);
            }
        } else {
            for (uint32_t i = 0; i < kr; ++i) {
                noc_async_read(x_acc.get_noc_addr(row * EMB_T + kstart + i), dst + i * BFP8_TILE, BFP8_TILE);
            }
        }
#else
        (void)row;
        (void)dst;
#endif
    };

    // True means the next cb_x_tiles slot was reserved during the previous block and this core's
    // injector row, if any, has already landed (bf16 sticks in cb_x_in; bfp8 tiles in the slot).
    bool x_prefetched = false;

    // `count == 0` -> m_blocks == 0 on every core: no CB traffic, no collective round, no
    // semaphore. Uniform across the grid, so it cannot hang.
    for (uint32_t b = 0; b < m_blocks; ++b) {
        // The RUNTIME token tile-rows this block actually works on. Identical on every core (it is
        // a pure function of the same mailbox words), which is what keeps the three collectives'
        // round counts and landing addresses in lockstep across the grid.
        const uint32_t m_eff = moe_fused_swiglu::m_tiles_eff(m_t, b, M_BLOCK, M_EFF_MIN);
        const bool wd_mrow = WD_MROW_ROUNDS && (m_eff == M_BLOCK);
        const uint32_t x_slot_tiles = m_eff * KR_PAD;   // resident in0 block, one slot
        const uint32_t h_block_tiles = m_eff * HN_PAD;  // one phase-2 K-block of h

        // The diagonal core in row r owns that row's HID_T-wide assembly slot.  Invite every
        // hidden-column worker only after the preceding block's phase 2 has consumed the slot.
        // The writer waits this monotone counter before depositing its reduced HN fragment.
        if (wd_mrow && is_row_agg) {
            const uint32_t sem = static_cast<uint32_t>(get_semaphore(SEM_HROW_FREE));
            for (uint32_t x = 0; x < HGROUPS; ++x) {
                const uint32_t sidx = my_row * HGROUPS + x;
                const uint32_t vx = get_arg_val<uint32_t>(RT_HMCAST + 4 + 2 * sidx + 0);
                const uint32_t vy = get_arg_val<uint32_t>(RT_HMCAST + 4 + 2 * sidx + 1);
                noc_semaphore_inc(get_noc_addr(vx, vy, sem), 1);
            }
            noc_async_atomic_barrier();
        }

        // REFINEMENT 3 — the weight DRAM read happens on M-block 0 only when the block is resident.
        // `cb_pop_front` advances a read pointer without touching the bytes, and each weight CB has
        // a single producer, so the slot a later M-block re-reserves still holds what block 0 read
        // into it. Everything else — reserve, push, barrier, trip counts — is unchanged, which is
        // what keeps compute bit-for-bit identical.
        const bool read_wg = (b == 0) || (W_RESIDENT == 0);
        const bool read_wd = (b == 0) || (WD_RESIDENT == 0);

        // ---- Phase 1a: stage x, then multicast it along the grid row ----
        // cb_x_tiles is ONE slot, so its write pointer is the same L1 address on every core in the
        // row — which mcast_pipe requires of the landing address. W_gate is issued before the x
        // chain and published after it as before; compute starts early on W_up, whose independent
        // writer stream is already live while these rounds run.
        auto issue_wg_chunk = [&](uint32_t c) {
            cb_reserve_back(cb_w_gate, WG_CHUNK_TILES);
            MaybeDeviceZoneScope("reader_wg_issue");
            const uint32_t wg_wp = get_write_ptr(cb_w_gate);
#ifndef ABLATE_NO_W_XFER  // /perf-measure: drop the weight DRAM stream, keep every CB + barrier
            moe_fused_swiglu::read_weight_chunk<BRG>(
                wg_acc, read_wg, c, GU_CHUNK_W, kr, kstart, hstart, hn, HID_T, wg_wp, W_TILE);
#else
            (void)wg_wp;
#endif
        };
        // WHERE the tail chunks are issued is the whole result: issuing all GU_CHUNKS here
        // measured +17 / +6 / +5 %, because the x staging prologue's blanket read barrier is
        // all-or-nothing and drained the whole weight block before a single stick was tilized.
        // Chunk 0 here; chunks 1..N-1 after that barrier. See DESIGN_NOTES.md §4.

        const bool staged_early = x_prefetched;
        x_prefetched = false;
        if (!staged_early) {
            cb_reserve_back(cb_x_tiles, x_slot_tiles);
        }
        const uint32_t x_base = get_write_ptr(cb_x_tiles);

        // ---- x staging PROLOGUE: land every tile-row THIS core injects, before the chain ----
        // Hoisted out of the multicast loop: staging is per-injector work with no cross-core
        // ordering, so doing it up front lets the chain run uninterrupted.
        {
            MaybeDeviceZoneScope("reader_xstage");
            // WHICH tile-row this column stages, and in WHAT ORDER its sticks are read. The walk
            // starts at `(my_col + my_row) % TILE_H` and wraps: the one rotation that moves which
            // BANK a core is on at a given instant, worth ~0.5 %. Rotating which COLUMN injects
            // instead was a measured null, twice. See DESIGN_NOTES.md §4.
            const uint32_t t_first = moe_fused_swiglu::inject_first(my_col);
            for (uint32_t t = t_first; t < m_eff; t += HGROUPS) {
                const uint32_t dst = x_base + t * X_ROW_BYTES;
                uint32_t row = b * M_BLOCK + t;
                if (row >= M_T_MAX) {
                    row = M_T_MAX - 1;  // rows past the sized region are UNDEFINED; stay in bounds
                }
                if constexpr (INPUT_FORMAT == 0) {
                    // bf16 ROW_MAJOR: read this row-group's emb slice of 32 sticks. Compute tilizes
                    // it DIRECTLY into `dst`; cb_x_ready is only a one-page completion channel, so
                    // the reader remains the sole owner of cb_x_tiles' push/write-pointer state.
                    if (!staged_early) {
                        cb_reserve_back(cb_x_in, TILE_H);
                        issue_x_row(row, get_write_ptr(cb_x_in));
                        noc_async_read_barrier();
                        cb_push_back(cb_x_in, TILE_H);
                    }

                    cb_wait_front(cb_x_ready, 1);
                    cb_pop_front(cb_x_ready, 1);
                } else {
                    // bfp8_b TILE: the tiles land straight in the resident slot, no tilize.
                    if (!staged_early) {
                        issue_x_row(row, dst);
                        noc_async_read_barrier();
                    }
                }
            }
        }

        // PERF 3 — this core's `x` is off DRAM. Release the writer's W_up stream (XPRIO). A plain
        // volatile store, not a NoC semaphore op: producer and consumer are two RISC-Vs on the SAME
        // core sharing one L1, and this word has exactly one writer. Monotone, so no reset.
        if constexpr (XPRIO) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_XSTAGED)) = b + 1;
        }

        // The staging prologue's barriers are behind us, so this prefetch has the multicast chain
        // and the up matmul to land under instead of standing in front of the stick reads.
        issue_wg_chunk(0);

        // ---- x multicast chain ----
        // m_eff rounds, not M_BLOCK: at count 128 (M_t 4) this is HALF the handshake chain and half
        // the staged bytes, and at count 32 an eighth. m_eff divides M_BLOCK, so cb_x_tiles' write
        // pointer stays block-aligned and identical on every core in the row (which mcast_pipe
        // requires of the landing address). A full M block stays wholly reserved but publishes each
        // completed row separately; smaller blocks retain the cheaper one-push handoff.
        {
            MaybeDeviceZoneScope("reader_xmcast");
            for (uint32_t t = 0; t < m_eff; ++t) {
#ifndef ABLATE_NO_X_XFER  // /perf-measure: drop the x transport, keep cb_x_tiles' reserve/push
                if constexpr (xmc.active) {
                    // PERF 13 — round `t` carries tile-row `t` (unchanged); only the LANE that
                    // sends it is skewed by the grid row, which is what turns a round's injector set
                    // from a column into a diagonal. Every core in the row derives the same lane
                    // from the same `my_row`, so sender and receivers agree by construction, and the
                    // lane value IS the sender's column so the coord table needs no change.
                    const uint32_t round = t % HGROUPS;
                    if (round == my_col) {
                        x_send.send(x_base + t * X_ROW_BYTES, x_base + t * X_ROW_BYTES, X_ROW_BYTES);
                    } else {
                        x_recv.receive(round);
                    }
                }
#endif
                if (m_eff == M_BLOCK) {
                    cb_push_back(cb_x_tiles, KR_PAD);
                }
            }
        }
        if (m_eff != M_BLOCK) {
            // At small M the multicast is already short, and per-row CB bookkeeping costs more
            // than it can hide. Preserve the original one-push handoff for those blocks.
            cb_push_back(cb_x_tiles, x_slot_tiles);
        }

        // ---- Phase 1b: W_gate landed under the x rounds; publish it ----
        // (W_up is the writer's twin on NoC1.) Publish chunk c, then issue c+1 and block on it, so
        // the reader sits in DRAM while compute chews c. Only chunk c is ever outstanding at a
        // barrier, so the all-or-nothing drain is exact here.
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
        // Phase 1b' — W_down for ALL WD_AHEAD phase-2 K-blocks, ISSUED as one batch, so the
        // reads land under the reduce rendezvous instead of in front of the round that needs them.
        constexpr uint32_t WD_BLOCK_TILES = HN_PAD * EC_MAX;
        constexpr bool CAN_PREFETCH_X = HGROUPS >= M_BLOCK;
        const bool prefetch_next_x = kPrefetchNextX && CAN_PREFETCH_X && (b + 1 < m_blocks);
        if (prefetch_next_x) {
            // Transaction id zero is the legacy/untagged stream and cannot be waited through the
            // scoped barrier on this architecture. Tag phase 2 before its first W_down issue.
            noc_async_read_set_trid(P2_READ_TRID);
        }
        auto issue_wd_batch = [&]() {
            const uint32_t nblocks = wd_mrow ? HGROUPS : WD_AHEAD;
            cb_reserve_back(cb_w_down, nblocks * WD_BLOCK_TILES);
            MaybeDeviceZoneScope("reader_wd_issue");
            const uint32_t wp = get_write_ptr(cb_w_down);
            (void)wp;
            for (uint32_t r = 0; r < nblocks; ++r) {
                const uint32_t hbase = r * HN_PAD;
                const uint32_t hn_r = moe_fused_swiglu::wd_block_rows(hbase, HN_PAD, HID_T);
#ifndef ABLATE_NO_W_XFER
                // The writer takes the TAIL rows on NOC_1; this is the head.
                if (read_wd) {
                    moe_fused_swiglu::read_wd_rows<BRD>(
                        wd_acc,
                        hbase,
                        0,
                        hn_r - wd_rows_writer(hn_r),
                        jstart,
                        ec,
                        EC_MAX,
                        EMB_T,
                        wp + r * WD_BLOCK_TILES * W_TILE,
                        W_TILE);
                }
#endif
            }
        };
        issue_wd_batch();

        // Start block b+1's activation read before block b's reduce + phase 2. At the supported
        // grids HGROUPS >= M_BLOCK, so each core injects at most one row and the existing one-row
        // cb_x_in is sufficient. Smaller grids retain the ordinary next-block prologue.
        bool prefetch_has_local_row = false;
        if (prefetch_next_x) {
            const uint32_t next_m_eff = moe_fused_swiglu::m_tiles_eff(m_t, b + 1, M_BLOCK, M_EFF_MIN);
            cb_reserve_back(cb_x_tiles, next_m_eff * KR_PAD);
            const uint32_t next_x_base = get_write_ptr(cb_x_tiles);
            const uint32_t t = moe_fused_swiglu::inject_first(my_col);
            if (t < next_m_eff) {
                prefetch_has_local_row = true;
                uint32_t row = (b + 1) * M_BLOCK + t;
                if (row >= M_T_MAX) {
                    row = M_T_MAX - 1;
                }
                noc_async_read_set_trid(NEXT_X_TRID);
                if constexpr (INPUT_FORMAT == 0) {
                    cb_reserve_back(cb_x_in, TILE_H);
                    issue_x_row(row, get_write_ptr(cb_x_in));
                } else {
                    issue_x_row(row, next_x_base + t * X_ROW_BYTES);
                }
                noc_async_read_set_trid(P2_READ_TRID);
            }
        }

        // ---- Phase 1c: the cross-column reduce, RECEIVER side ----
        // SEM_GO is the invite — the flow control that stops a contributor overwriting a landing
        // slot compute has not consumed. MY SLICE comes from the ONE shared plan
        // (moe_fused_swiglu_common.hpp): `slice_workers` cores own `slice_tiles` each, identical
        // on every core and every RISC-V.
        const uint32_t sl_w = moe_fused_swiglu::slice_workers(h_block_tiles, KGROUPS);
        const uint32_t sl_a = (sl_w != 0) ? (h_block_tiles / sl_w) : 0;  // uniform slice size
        const uint32_t slice_tiles = (my_row < sl_w) ? sl_a : 0;         // 0 = an idle core
        MaybeDeviceZoneScope("reader_reduce");
        // Reserve the landing CBs WHOLE first, THEN invite the whole column, then wait for every
        // contributor and push WHOLE. This is also cb_h_local's flow control, transitively: my
        // invite for block b+1 is issued after my phase 2 of block b has read it, and no worker's
        // h-slice send for b+1 can precede this invite. So the h landing needs no second handshake.
        if (slice_tiles) {
            cb_reserve_back(cb_gather_gate, GATHER_PAGES);
            cb_reserve_back(cb_gather_up, GATHER_PAGES);
        }
        // cb_gather_gate aliases cb_h_slice and cb_out_tiles for BFP8 output. Reserving changes
        // only this logical view's FIFO state, but the invite below authorises PEERS to write its
        // physical SRAM. Block b therefore cannot send the invite until this core's writer has
        // drained block b-1's output DMA from that same SRAM. This target-local edge is required:
        // a peer's own writer progress says nothing about this core's outstanding output read.
        if constexpr (PHASE_CB_ALIAS) {
            if (b != 0) {
                moe_fused_swiglu::sem_wait_min(SEM_PHASE_FREE, b);
            }
        }
#ifndef ABLATE_NO_REDUCE_XFER  // /perf-measure: drop the all-to-all, keep every CB cycle
        const uint32_t sem_go = static_cast<uint32_t>(get_semaphore(SEM_GO));
        for (uint32_t i = 0; i < KGROUPS; ++i) {
            const uint32_t p = i;
            const uint32_t px = get_arg_val<uint32_t>(RT_PEERS + 2 * p + 0);
            const uint32_t py = get_arg_val<uint32_t>(RT_PEERS + 2 * p + 1);
            noc_semaphore_inc(get_noc_addr(px, py, sem_go), 1);
        }
        noc_async_atomic_barrier();
#endif
        {
            cb_wait_front(cb_up_acc, h_block_tiles);
#ifndef ABLATE_NO_REDUCE_XFER
            // The UP half of the column all-to-all, on NOC_0; the writer carries the GATE half.
            // Wait for the WHOLE column's invites first, exactly as the writer does: every core
            // invites once per peer per M-block, so (b+1)*KGROUPS is the exact total.
            moe_fused_swiglu::sem_wait_min(SEM_GO, (b + 1) * KGROUPS);
            moe_fused_swiglu::scatter_leg(RT_PEERS, cb_up_acc, cb_gather_up, SEM_DATA, sl_w, sl_a, my_row, BFP8_TILE);
#endif
            cb_pop_front(cb_up_acc, h_block_tiles);
        }
        if (slice_tiles) {
#ifndef ABLATE_NO_REDUCE_XFER
            // TWO signals per contributor — one per payload, since gate and up ride different
            // NoCs. The total is what says "everything landed".
            data_arrivals += 2 * KGROUPS;
            noc_semaphore_wait_min(sem_data_ptr, data_arrivals);
#endif
            cb_push_back(cb_gather_gate, GATHER_PAGES);
            cb_push_back(cb_gather_up, GATHER_PAGES);
        }

        // ---- Phase 2: broadcast h and consume W_down ----
        // Full M-blocks run one whole-H broadcast per M row against the complete resident W_down
        // shard. Ragged blocks retain the hidden-slice rounds and their deferred read barrier:
        // each W_down read lands under the next grid-wide multicast rather than being drained on
        // the spot. `wd_pending` carries that issued-but-unpublished block between rounds.
        {
            MaybeDeviceZoneScope("reader_wd_wait");
            if (prefetch_next_x) {
                noc_async_read_barrier_with_trid(P2_READ_TRID);
            } else {
                noc_async_read_barrier();
            }
            if constexpr (WD_SPLIT) {
                // ...and NOC_1's half of the SAME blocks (see wd_split_gate).
                wd_split_gate(wd_done, wd_mrow ? HGROUPS : WD_AHEAD);
            }
        }
        cb_push_back(cb_w_down, (wd_mrow ? HGROUPS : WD_AHEAD) * WD_BLOCK_TILES);
        // PERF 2 — on the scatter path this column's h block is ASSEMBLED IN cb_h_local BY THE
        // WORKERS' NoC WRITES, not packed by compute, so the root's handshake is the SEM_HSLICE
        // counter rather than a CB front. `sl_w` slices land per M-block; the counter is monotone and
        // cumulative, like every other semaphore in this op.
        if (wd_mrow) {
            if (is_row_agg) {
                h_arrivals += HGROUPS;
            }
        } else if (is_root) {
            h_arrivals += sl_w;
        }
        if (prefetch_next_x) {
            // Every read issued from here to the end of phase 2 is either a W_down head or the
            // sender's local h copy. Scoped barriers may drain these without touching NEXT_X_TRID.
            noc_async_read_set_trid(P2_READ_TRID);
        }
        auto phase2_read_barrier = [&]() {
            if (prefetch_next_x) {
                noc_async_read_barrier_with_trid(P2_READ_TRID);
            } else {
                noc_async_read_barrier();
            }
        };
        {
            MaybeDeviceZoneScope("reader_phase2");
            if (wd_mrow) {
                // Eight M-row rounds.  Row r's diagonal aggregator owns one complete HID_T-wide
                // activation row; broadcast it as a single K=HID_T operand.  W_down's complete
                // resident weight shard was published above, so there is no per-round weight DMA.
                uint32_t next_ack = 0;
                for (uint32_t r = 0; r < KGROUPS; ++r) {
                    uint32_t ahead = HACK_AHEAD;
                    if (ahead > DEPTH_H) {
                        ahead = DEPTH_H;
                    }
                    if (ahead > KGROUPS - r) {
                        ahead = KGROUPS - r;
                    }
                    cb_reserve_back(cb_h, ahead * HID_T);
                    const uint32_t hdst = get_write_ptr(cb_h);
                    while (next_ack < r + ahead && next_ack < KGROUPS) {
                        const uint32_t sidx = next_ack * HGROUPS + next_ack;
                        const uint32_t svx = get_arg_val<uint32_t>(RT_HMCAST + 4 + 2 * sidx + 0);
                        const uint32_t svy = get_arg_val<uint32_t>(RT_HMCAST + 4 + 2 * sidx + 1);
                        noc_semaphore_inc(get_noc_addr(svx, svy, static_cast<uint32_t>(get_semaphore(SEM_H_FREE))), 1);
                        ++next_ack;
                    }

                    const bool i_send = is_row_agg && (my_row == r);
                    const uint32_t slot = (b * KGROUPS + r) % DEPTH_H;
                    if (i_send) {
#ifndef ABLATE_NO_REDUCE_XFER
                        noc_semaphore_wait_min(sem_h_ptr, h_arrivals);
#endif
                        noc_async_read(get_noc_addr(get_write_ptr(cb_h_local)), hdst, HID_T * H_TILE);
                        phase2_read_barrier();
#ifndef ABLATE_NO_H_XFER
                        if constexpr (hmc.active) {
                            h_free_expected += NUM_CORES;
                            noc_semaphore_wait_min(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                    static_cast<uint32_t>(get_semaphore(SEM_H_FREE))),
                                h_free_expected);
                            h_slot_send_posted(slot, hdst, HID_T * H_TILE);
                        }
#endif
                    } else {
#ifndef ABLATE_NO_H_XFER
                        if constexpr (hmc.active) {
                            volatile tt_l1_ptr uint32_t* hf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                static_cast<uint32_t>(get_semaphore(SEM_H_RDY_BASE + slot)));
                            MaybeDeviceZoneScope("p2_hwait");
                            noc_semaphore_wait(hf, VALID);
                            noc_semaphore_set(hf, INVALID);
                        }
#endif
                    }
                    cb_push_back(cb_h, HID_T);
                }
                // Eight 64-tile pushes advance a 3x64-tile CB by two slots.  Publish one payload-
                // free slot so both producer and consumer return to the CB base before a possible
                // smaller tail block switches back to the HN_PAD-wide schedule.
                cb_reserve_back(cb_h, HID_T);
                cb_push_back(cb_h, HID_T);
            } else {
                bool wd_pending = false;
                // HACK_AHEAD — how many rounds' senders this core acks in one go. The all-gather costs
                // 3.12 us of FIXED per-round rendezvous against 2.06 us of work, so acking ahead is the
                // round-cost lever; legal only because the VALID cells are per-slot. The real bound is
                // runtime (`blocks_cap = DEPTH_H * M_BLOCK / m_eff`) because m_eff is, so it is clamped
                // here rather than host-side. See DESIGN_NOTES.md §3.
                const uint32_t blocks_cap = (DEPTH_H * M_BLOCK) / m_eff;
                uint32_t hack_ahead = HACK_AHEAD;
                if (hack_ahead > blocks_cap - 1) {
                    hack_ahead = blocks_cap - 1;  // never reserve the WHOLE CB: that forces 0 in flight
                }
                // ...and never ack more than DEPTH_H ahead, whatever the CB slack. There are only
                // DEPTH_H flag cells, so sender g + A - 1 reuses the cell last written by global round
                // g + A - 1 - DEPTH_H; only A <= DEPTH_H keeps that strictly in this core's PAST, i.e.
                // already waited on and put back to INVALID. blocks_cap alone is NOT this bound (it is
                // 6 at m_eff 4 and 24 at m_eff 1) and using it hangs.
                if (hack_ahead > DEPTH_H) {
                    hack_ahead = DEPTH_H;
                }
                if (hack_ahead < 1) {
                    hack_ahead = 1;
                }
                uint32_t next_ack = 0;
                for (uint32_t r = 0; r < HGROUPS; ++r) {
                    // Reserve this round's cb_h slot first so the sender can issue its self-copy
                    // before the barrier below, which then covers the self-copy AND the previous
                    // round's W_down block in one drain. The tail clamp (`HGROUPS - r`) matters:
                    // near the end, reserving blocks nobody will push into would hang here.
                    {
                        // Per-round zones: 3 records/round = 33/M-block on top of the reader's 8,
                        // against a 125-per-core cap — so these resolve ONE M-block (count <= 256) and
                        // only the whole-kernel duration above that.
                        MaybeDeviceZoneScope("p2_reserve");
                        uint32_t ahead = hack_ahead;
                        if (ahead > HGROUPS - r) {
                            ahead = HGROUPS - r;
                        }
                        cb_reserve_back(cb_h, ahead * h_block_tiles);
                    }
                    const uint32_t hdst = get_write_ptr(cb_h);
                    // ACK FIRST, then wait. `cb_reserve_back` above proves THIS core's slot is free,
                    // so tell round r's sender it may write before blocking on round r's arrival.
                    // Acking first is what makes this deadlock-free: every sender's window ack comes
                    // from a core that has already reserved, so no core waits on a core waiting on it.
                    while (next_ack < r + hack_ahead && next_ack < HGROUPS) {
                        const uint32_t sidx = (next_ack % KGROUPS) * HGROUPS + next_ack;
                        const uint32_t svx = get_arg_val<uint32_t>(RT_HMCAST + 4 + 2 * sidx + 0);
                        const uint32_t svy = get_arg_val<uint32_t>(RT_HMCAST + 4 + 2 * sidx + 1);
                        noc_semaphore_inc(get_noc_addr(svx, svy, static_cast<uint32_t>(get_semaphore(SEM_H_FREE))), 1);
                        ++next_ack;
                    }
                    const bool i_send = (is_root && r == my_col);
                    if (i_send) {
                        // Self-copy cb_h_local -> this round's cb_h slot, so the send below is
                        // `src == dst` and therefore EXCLUDE-source. A src != dst send is a LOOPBACK
                        // multicast whose rotating-sender flag reset races this core's own in-flight
                        // VALID: measured as PCC drifting 0.959-0.979 run to run on a fixed input.
                        // cb_h_local needs no CB front — the workers' NoC writes assemble it.
#ifndef ABLATE_NO_REDUCE_XFER  // the workers' sends are stubbed too, so this wait must go with them
                    noc_semaphore_wait_min(sem_h_ptr, h_arrivals);
#endif
                    noc_async_read(get_noc_addr(get_write_ptr(cb_h_local)), hdst, h_block_tiles * H_TILE);
                    }

                if (i_send) {
                    // The SENDER must drain before it broadcasts (its self-copy has to have landed), so
                    // it also publishes the pending W_down block here. One core per round pays this.
                    phase2_read_barrier();  // the self-copy AND the previous round's W_down block
                    if (wd_pending) {
                        if constexpr (WD_SPLIT) {
                            wd_split_gate(wd_done, 1);
                        }
                        cb_push_back(cb_w_down, WD_BLOCK_TILES);
                        wd_pending = false;
                    }
#ifndef ABLATE_NO_H_XFER  // /perf-measure: drop the h transport, keep cb_h's reserve/push
                    // PER-SLOT FLAGS. Linked data+signal multicast, so still NO acked write
                    // barrier, and the VALID cell is this SLOT's — round r+1's sender is not held
                    // behind every core clearing round r's. The ack accounting is the MONOTONE
                    // `h_free_expected` counter, because HACK_AHEAD deliberately breaks the
                    // round-to-round chain a reset-based handshake would need.
                    if constexpr (hmc.active) {
                        h_free_expected += NUM_CORES;
                        noc_semaphore_wait_min(
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                static_cast<uint32_t>(get_semaphore(SEM_H_FREE))),
                            h_free_expected);
                        // `src == dst`, so exclude-source: a src != dst send is a LOOPBACK
                        // multicast whose flag reset races this core's own in-flight VALID.
                        h_slot_send_posted((b * HGROUPS + r) % DEPTH_H, hdst, h_block_tiles * H_TILE);
                    }
#endif
                } else {
                    // Every core but this round's sender drains AFTER the multicast, so the W_down read
                    // had this whole round's grid-wide broadcast to land under — that is the deferral.
#ifndef ABLATE_NO_H_XFER
                    // The whole of `ReceiverPipe::receive` for a Flag signal with PRE_HANDSHAKE
                    // off: wait for THIS slot's VALID, then put it back. Raw rather than a per-slot
                    // ReceiverPipe because that class's ctor sets the cell INVALID and would clobber
                    // a VALID a sender running ahead had already broadcast.
                    if constexpr (hmc.active) {
                        volatile tt_l1_ptr uint32_t* hf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            static_cast<uint32_t>(get_semaphore(SEM_H_RDY_BASE + ((b * HGROUPS + r) % DEPTH_H))));
                        MaybeDeviceZoneScope("p2_hwait");
                        noc_semaphore_wait(hf, VALID);
                        noc_semaphore_set(hf, INVALID);
                    }
#endif
                    if (wd_pending) {
                        // PERF 15 — the REAL per-round W_down wait. `reader_wd_wait` covers only the
                        // WD_AHEAD=1 prologue block (1 of 11) and I misread it as "down never waits
                        // for its weights"; this is every other round, on the non-sending cores.
                        MaybeDeviceZoneScope("p2_wdbar");
                        phase2_read_barrier();
                        if constexpr (WD_SPLIT) {
                            wd_split_gate(wd_done, 1);
                        }
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
                    const uint32_t hn_r = moe_fused_swiglu::wd_block_rows(hbase, HN_PAD, HID_T);
                    cb_reserve_back(cb_w_down, WD_BLOCK_TILES);
                    const uint32_t wp = get_write_ptr(cb_w_down);
#ifndef ABLATE_NO_W_XFER
                    // The writer already has the tail rows in flight on NOC_1 (it issued every
                    // K-block as one batch back in phase 1), so this reads only the head.
                    if (read_wd) {
                        moe_fused_swiglu::read_wd_rows<BRD>(
                            wd_acc, hbase, 0, hn_r - wd_rows_writer(hn_r), jstart, ec, EC_MAX, EMB_T, wp, W_TILE);
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
                phase2_read_barrier();
                if constexpr (WD_SPLIT) {
                    wd_split_gate(wd_done, 1);
                }
                cb_push_back(cb_w_down, WD_BLOCK_TILES);
            }
            }
        }
        if (prefetch_next_x) {
            noc_async_read_set_trid(0);
            noc_async_read_barrier_with_trid(NEXT_X_TRID);
            if constexpr (INPUT_FORMAT == 0) {
                if (prefetch_has_local_row) {
                    cb_push_back(cb_x_in, TILE_H);
                }
            }
            x_prefetched = true;
        }
    }
}
