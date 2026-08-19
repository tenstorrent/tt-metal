// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
//     transaction-rate-bound rather than bandwidth-bound. Addresses still go via TensorAccessor.
//   * the reduce-scatter transport is raw unicast + counting semaphores: mcast_pipe's SenderPipe is
//     a rectangle multicast, while a gather leg is point-to-point with a different destination per
//     peer, and the fan-in needs counting.
//   * the h multicast is raw because `noc.h` blocks POSTED multicast at the library level.
//   * the token-count publish is a raw L1 mailbox because compute needs a scalar loop bound on ALL
//     THREE TRISCs and `cb_wait_front` in a compute kernel is UNPACK-only.
//
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/assert.h"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"

#include "moe_fused_swiglu_dataflow.hpp"  // the transport vocabulary shared with the writer
#include "moe_fused_swiglu_common.hpp"    // the ONE definition of the mailbox word layout
#include "moe_fused_swiglu_ct_args.hpp"   // the ONE definition of the compile-time arg order

// Keep profiling source-compatible with the operation's detailed zones without depending on
// kernel_lib's convenience wrapper. Ordinary profiler sweeps leave these off so stage-record
// traffic does not alter the latency being measured.
#ifdef MOE_FUSED_SWIGLU_STAGE_PROFILE
#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)
#else
#define MaybeDeviceZoneScope(name)
#endif

// Set MOE_FUSED_SWIGLU_STAGE_PROFILE=1 before process start for detailed bottleneck runs. Budget:
// 8 records per M-block against a 125-per-core cap, so a run resolves stages for m_blocks <= 15.

// Compile-time block model. Every trip count and CB increment below is derived
// from these; none is a literal.
MOE_DECLARE_CT_ENUM(MOE_READER_CT_ARGS);

constexpr uint32_t INPUT_FORMAT = CT(INPUT_FORMAT);  // 0 = bf16 RM sticks, 1 = bfp8 tiles
constexpr uint32_t M_T_MAX = CT(M_T_MAX);
constexpr uint32_t LOCAL_EXPERT_ID = CT(LOCAL_EXPERT_ID);
constexpr uint32_t EMB_T = CT(EMB_T);
constexpr uint32_t HID_T = CT(HID_T);
constexpr uint32_t KR_PAD = CT(KR_PAD);              // K tiles per row-group slot (uniform)
constexpr uint32_t HN_PAD = CT(HN_PAD);              // hidden tiles per column-group (uniform)
constexpr uint32_t EC_MAX = CT(EC_MAX);              // phase-2 N stride (uniform CB increment)
constexpr uint32_t WD_EC_MAX = CT(WD_EC_MAX);        // resident W_down row stride in both modes
constexpr uint32_t EC_GROUP_MAX = CT(EC_GROUP_MAX);  // output stride in the paired-row mode
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
constexpr uint32_t H_ROUND_NOC1_MASK = CT(H_ROUND_NOC1_MASK);
constexpr uint32_t SCATTER_ONE_SIGNAL = CT(SCATTER_ONE_SIGNAL);

// X_PAGE is the ACTIVATION TENSOR's own page (bf16: one full emb stick; bfp8: one tile) — what
// TensorAccessor needs to place a page in a bank. X_SLICE is the cb_x_in page stride, i.e. only
// this row-group's KR_PAD-tile slice of a stick. Not the same number.
constexpr uint32_t X_PAGE = CT(X_PAGE);
constexpr uint32_t X_SLICE = CT(X_SLICE);
constexpr uint32_t COUNTS_PAGE = CT(COUNTS_PAGE);
constexpr uint32_t IDX_PAGE = CT(IDX_PAGE);
constexpr uint32_t W_TILE = CT(W_TILE_BYTES);  // weight tile stride: bfp4 576, bfp8 1088, bf16 2048
constexpr uint32_t BFP8_TILE = CT(BFP8_TILE);
// h is bfp8, like x, the output and the reduce operands. It cannot be bfp4: the packer emits bfp8,
// so a bfp4 h CB would be decoded through the wrong format.
constexpr uint32_t H_TILE = BFP8_TILE;
constexpr uint32_t MAILBOX_MAGIC = CT(MAILBOX_MAGIC);

// W_down blocks kept in flight ahead of the round that consumes them; 1 == the per-round read.
constexpr uint32_t WD_AHEAD = CT(WD_AHEAD);
// Smallest legal `m_eff`. One host definition, identical in all three kernels — see m_tiles_eff().
constexpr uint32_t M_EFF_MIN = CT(M_EFF_MIN);
// Cross-M-block weight residency: every weight read is a pure function of this core's
// kstart/hstart/out_col_start with no M-block index, so block_idx > 0 re-reads bytes still in the CB slot. The
// reserve/push handshake is untouched; only the DRAM read loops are skipped.
constexpr uint32_t W_RESIDENT = CT(W_RESIDENT);
constexpr uint32_t WD_RESIDENT = CT(WD_RESIDENT);
constexpr uint32_t WD_MROW_ROUNDS = CT(WD_MROW_ROUNDS);
constexpr uint32_t WD_MGROUPS = CT(WD_MGROUPS);
constexpr uint32_t WD_MGROUP_MIN_BLOCKS = CT(WD_MGROUP_MIN_BLOCKS);
constexpr uint32_t MGROUP_ROWS = CT(MGROUP_ROWS);
constexpr uint32_t MGROUP_CORES = HGROUPS * MGROUP_ROWS;
constexpr bool WD_PACKED = WD_RESIDENT && moe_fused_swiglu::hidden_blocks_are_balanced(HID_T, HGROUPS, HN_PAD);
constexpr uint32_t HROW_T = HID_T;
constexpr uint32_t GU_CHUNKS = CT(GU_CHUNKS);
constexpr uint32_t XPRIO = CT(XPRIO);
constexpr uint32_t HACK_AHEAD = CT(HACK_AHEAD);
constexpr uint32_t DEPTH_H = CT(DEPTH_H);
constexpr uint32_t DEPTH_X = CT(DEPTH_X);
constexpr uint32_t WD_SPLIT = CT(WD_SPLIT);
constexpr uint32_t WG_SHARD_W = CT(WG_SHARD_W);
constexpr uint32_t WD_SHARD_W = CT(WD_SHARD_W);
constexpr uint32_t GATHER_PAGES = CT(GATHER_PAGES);  // the WHOLE landing CB, in tiles

// SHARED-BUFFER REGION MODE (fused extract / insert). NEED_START: fetch start[global_expert_id] and
// publish it to the mailbox — set when EITHER side is on, since the writer's half arrives that way.
// READ_X_AT_OFFSET: this expert's tokens begin at start[global_id], not row 0.
constexpr uint32_t NEED_START = CT(NEED_START);
constexpr uint32_t READ_X_AT_OFFSET = CT(READ_X_AT_OFFSET);
constexpr uint32_t START_PAGE = CT(START_PAGE);

constexpr uint32_t cb_x_in = CT(CB_X_IN);
constexpr uint32_t cb_x_tiles = CT(CB_X_TILES);
constexpr uint32_t cb_tilize_done = CT(CB_X_STAGE);  // the compute->reader per-row completion edge
constexpr uint32_t cb_w_gate = CT(CB_W_GATE);
constexpr uint32_t cb_w_down = CT(CB_W_DOWN);
constexpr uint32_t cb_h = CT(CB_H);
constexpr uint32_t cb_h_local = CT(CB_H_LOCAL);
constexpr uint32_t cb_idx_scratch = CT(CB_IDX_SCRATCH);
constexpr uint32_t cb_counts_scratch = CT(CB_COUNTS_SCRATCH);
constexpr uint32_t cb_gather_gate = CT(CB_GATHER_GATE);
constexpr uint32_t cb_gather_up = CT(CB_GATHER_UP);
constexpr uint32_t cb_up_acc = CT(CB_UP_ACC);
constexpr uint32_t cb_mailbox_compute = CT(CB_MAILBOX_COMPUTE);
constexpr uint32_t cb_mailbox_writer = CT(CB_MAILBOX_WRITER);

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

// Runtime-arg layout: the 17-word scalar block, then the COLUMN as KGROUPS (vx, vy) pairs in ROW
// order — the invite fan-out and up-gather destinations. Row r at index r on every core is what
// makes the slice ownership agree grid-wide.
constexpr uint32_t RT_PEERS = 17;
constexpr uint32_t RT_XMCAST = RT_PEERS + 2 * KGROUPS;
constexpr uint32_t RT_HMCAST = RT_XMCAST + 4 + 2 * HGROUPS;

// The host emits the same five-word mcast descriptor used by kernel_lib:
// active, data-ready semaphore, consumer-ready semaphore, active consumers,
// flags.  This kernel only uses the rotating FLAG + pre-handshake x path.
constexpr bool XMCAST_ACTIVE = get_compile_time_arg_val(CT_XMCAST + 0) != 0;
constexpr uint32_t XMCAST_READY_SEM = get_compile_time_arg_val(CT_XMCAST + 1);
constexpr uint32_t XMCAST_FREE_SEM = get_compile_time_arg_val(CT_XMCAST + 2);
constexpr uint32_t XMCAST_CONSUMERS = get_compile_time_arg_val(CT_XMCAST + 3);
constexpr bool HMCAST_ACTIVE = get_compile_time_arg_val(CT_HMCAST + 0) != 0;
// h's descriptor has a rotating sender list of the whole grid.  The grouped
// rectangle follows that list; only the rectangles are used by this raw path.
constexpr uint32_t RT_HGROUP_RECT = RT_HMCAST + 4 + 2 * HGROUPS * KGROUPS;

// POSTED (default) drops the NUM_CORES-1 payload write-acks and changes nothing else: the VALID
// flag stays non-posted and LINKED on the same VC, so it cannot overtake the payload. Keep 0
// reachable — if that ordering ever fails a receiver reads a half-written slot, silently.
constexpr bool kHMcastPosted = (H_MCAST_POSTED != 0);

inline bool h_round_on_writer(uint32_t r) { return ((H_ROUND_NOC1_MASK >> r) & 1u) != 0; }

inline void h_slot_send_posted(uint32_t slot, uint32_t l1, uint32_t size, bool grouped = false) {
    const auto hrect = grouped ? moe_fused_swiglu::McastRect<noc_index>(
                                     get_arg_val<uint32_t>(RT_HGROUP_RECT + 0),
                                     get_arg_val<uint32_t>(RT_HGROUP_RECT + 1),
                                     get_arg_val<uint32_t>(RT_HGROUP_RECT + 2),
                                     get_arg_val<uint32_t>(RT_HGROUP_RECT + 3))
                               : moe_fused_swiglu::McastRect<noc_index>(
                                     get_arg_val<uint32_t>(RT_HMCAST + 0),
                                     get_arg_val<uint32_t>(RT_HMCAST + 1),
                                     get_arg_val<uint32_t>(RT_HMCAST + 2),
                                     get_arg_val<uint32_t>(RT_HMCAST + 3));
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
// `start` (= expert_region_offsets), appended LAST so adding it shifted no existing accessor. The
// host always emits this block — it stands `counts` in when the caller passes no offsets tensor —
// so the arg stream has one length and the accessor is simply unread when NEED_START is 0.
constexpr auto start_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();

// Three `WeightRuns` bindings, because the three tensors this kernel touches can have DIFFERENT
// placements: each weight stream takes its own tensor's DRAM ND shard width (0 = interleaved,
// one transaction per tile), and everything else stays on the interleaved binding.
using BR = moe_fused_swiglu::WeightRuns<>;
using BRG = moe_fused_swiglu::WeightRuns<WG_SHARD_W>;  // W_gate
using BRD = moe_fused_swiglu::WeightRuns<WD_SHARD_W>;  // W_down

// The W_down NoC split: WD_SPLIT eighths of every phase-2 K-block's hidden rows are read by the
// WRITER on NOC_1; this kernel keeps the head rows. The writer takes the TAIL rows so both sides
// read a contiguous run and the coalescing is unchanged on either side.
inline uint32_t wd_rows_writer(uint32_t block_hn_rows) { return (block_hn_rows * WD_SPLIT) / 8; }

// Cross-RISC completion gate. `noc_async_read_barrier()` is PER-RISC-V, so it proves nothing about
// the writer's share of the SAME K-blocks — publishing without this hands `down` a half-written
// tile. Both counters are monotone, so `wd_done + n` is the tightest legal gate.
inline void wd_split_gate(uint32_t& wd_done, uint32_t n) {
    wd_done += n;
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_WDSPLIT)), wd_done);
}

void kernel_main() {
    (void)get_arg_val<uint32_t>(0);  // retained runtime slot for cache-compatible argument layout
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_gate_addr = get_arg_val<uint32_t>(2);
    const uint32_t w_down_addr = get_arg_val<uint32_t>(3);
    const uint32_t counts_addr = get_arg_val<uint32_t>(4);
    const uint32_t idx_addr = get_arg_val<uint32_t>(5);
    const uint32_t kr_rows = get_arg_val<uint32_t>(6);  // real K tiles this grid ROW owns
    const uint32_t kstart = get_arg_val<uint32_t>(7);   // first emb tile index this row owns
    const uint32_t hstart = get_arg_val<uint32_t>(8);   // first hidden linear index this COLUMN owns
    const uint32_t hn_cols = get_arg_val<uint32_t>(9);  // real hidden tiles this column owns
    const uint32_t ec = get_arg_val<uint32_t>(10);      // output emb tiles this CORE owns
    const uint32_t out_col_start = get_arg_val<uint32_t>(11);
    const uint32_t ec_group = get_arg_val<uint32_t>(12);  // paired-row output ownership
    const uint32_t jstart_group = get_arg_val<uint32_t>(13);
    const uint32_t my_col = get_arg_val<uint32_t>(14);
    // My row in the grid column: which slice of the reduce-scatter I own (0 tiles = an idle core,
    // which still contributes and still invites).
    const uint32_t my_row = get_arg_val<uint32_t>(15);
    // `start` (= expert_region_offsets) base. Present in every dispatch (the host stands `counts`
    // in when the caller passes no offsets tensor) and read only when NEED_START.
    const uint32_t start_addr = get_arg_val<uint32_t>(16);
    // Column `x`'s reduce root is row `x % KGROUPS` — the core that injects this column's h into
    // the phase-2 all-gather. Derived, not passed: one rule, three kernels.
    const bool is_root = (my_row == my_col % KGROUPS);
    const bool is_row_agg = (my_col == my_row);

    const auto x_acc = TensorAccessor(x_args, x_addr, X_PAGE);
    const auto wg_acc = TensorAccessor(wg_args, w_gate_addr, W_TILE);
    const auto wd_acc = TensorAccessor(wd_args, w_down_addr, W_TILE);
    const auto cnt_acc = TensorAccessor(cnt_args, counts_addr, COUNTS_PAGE);
    const auto idx_acc = TensorAccessor(idx_args, idx_addr, IDX_PAGE);
    const auto start_acc = TensorAccessor(start_args, start_addr, START_PAGE);
    const uint32_t wd_base = get_write_ptr(cb_w_down);

    // The compute and writer mailbox CBs have independent FIFO state but alias CB_X_STAGE's 64 B
    // allocation. CB_X_STAGE itself is NOT published here — it stays the compute->reader per-row
    // tilization edge, so its event cannot be consumed by two readers at startup.
    cb_reserve_back(cb_mailbox_compute, 1);
    cb_reserve_back(cb_mailbox_writer, 1);
    const uint32_t mailbox_addr = get_write_ptr(cb_mailbox_compute);
    ASSERT(mailbox_addr == get_write_ptr(cb_tilize_done));
    ASSERT(mailbox_addr == get_write_ptr(cb_mailbox_writer));
    volatile tt_l1_ptr uint32_t* mailbox_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_addr);
    mailbox_words[moe_fused_swiglu::MBOX_HSEND_DONE] = 0;
    mailbox_words[moe_fused_swiglu::MBOX_UP_SCATTER_DONE] = 0;

    // Phase 0 — the device-resident count. count = counts[ idx[local_expert_id] ].
    // Two one-page reads into unpushed scratch CBs, read back through a volatile L1 pointer.
    const uint32_t l1_idx = get_write_ptr(cb_idx_scratch);
    const uint32_t l1_cnt = get_write_ptr(cb_counts_scratch);
    // Both accessors fetch page zero into distinct scratch pages.  Only the L1 lookup below
    // depends on `global_expert_id`; the counts PAGE address does not, so issue the two independent DRAM reads
    // together and pay one completion round-trip.  The optional region-start read remains later:
    // it deliberately reuses l1_cnt after `count` is extracted.
    noc_async_read(idx_acc.get_noc_addr(0), l1_idx, IDX_PAGE);
    noc_async_read(cnt_acc.get_noc_addr(0), l1_cnt, COUNTS_PAGE);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t global_expert_id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_idx)[LOCAL_EXPERT_ID];
    const uint32_t count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_cnt)[global_expert_id];

    uint32_t m_t = (count + TILE_H - 1) / TILE_H;
    if (m_t > M_T_MAX) {
        m_t = M_T_MAX;
    }
    const uint32_t m_blocks = (m_t + M_BLOCK - 1) / M_BLOCK;
    // One dispatch owns one resident W_down payload layout. Group only when every block is full,
    // so a ragged tail can never switch out_col_start/ec underneath the weights loaded at block_idx == 0.
    const bool wd_mgroup = WD_MGROUPS && (m_blocks >= WD_MGROUP_MIN_BLOCKS) && (m_t != 0) && ((m_t % M_BLOCK) == 0);
    const uint32_t wd_out_width = wd_mgroup ? ec_group : ec;
    const uint32_t wd_jstart = wd_mgroup ? jstart_group : out_col_start;

    // This expert's REGION BASE in a shared buffer, in token rows. Reuses cb_counts_scratch's page,
    // dead once `count` is extracted and exactly the right size (host validates equal lengths), so
    // the fused mode costs zero extra L1.
    uint32_t start_row = 0;
    if constexpr (NEED_START) {
        const uint32_t l1_start = get_write_ptr(cb_counts_scratch);
        noc_async_read(start_acc.get_noc_addr(0), l1_start, START_PAGE);
        noc_async_read_barrier();
        invalidate_l1_cache();
        start_row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_start)[global_expert_id];
        // Region bases are tile-aligned by construction (the dispatch lays experts out in whole
        // tile-rows) and every consumer floors by TILE_H, so a misaligned base would silently
        // shift this expert's rows rather than fail.
        ASSERT(start_row % TILE_H == 0);
    }

    // Publish {count, M_t, m_blocks, start_row} so compute (all three TRISCs) and the writer can
    // read it. The writer's half of the fusion arrives ONLY through this word.
    moe_fused_swiglu::mailbox_publish(mailbox_addr, MAILBOX_MAGIC, count, m_t, m_blocks, start_row);
    cb_push_back(cb_mailbox_compute, 1);
    cb_push_back(cb_mailbox_writer, 1);
    // x-read rebase, derived once. Row-major x is addressed by STICK (one page per token row), so
    // the offset is the token row itself; tiled x is addressed by TILE PAGE at an EMB_T row stride.
    const uint32_t x_stick_base = READ_X_AT_OFFSET ? start_row : 0;
    const uint32_t x_tile_base = READ_X_AT_OFFSET ? (start_row / TILE_H) * EMB_T : 0;

    // Row multicast state.  All receivers initialize their own flag before
    // acknowledging a sender; the sender waits for every acknowledgement, so
    // this is the same happens-before edge as the former SenderPipe/ReceiverPipe.
    const auto xrect = moe_fused_swiglu::McastRect<noc_index>(
        get_arg_val<uint32_t>(RT_XMCAST + 0),
        get_arg_val<uint32_t>(RT_XMCAST + 1),
        get_arg_val<uint32_t>(RT_XMCAST + 2),
        get_arg_val<uint32_t>(RT_XMCAST + 3));
    const auto& xbounds = xrect.bounds();
    const uint32_t x_mcast_dests = xrect.area() - 1;
    volatile tt_l1_ptr uint32_t* x_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(XMCAST_READY_SEM)));
    volatile tt_l1_ptr uint32_t* x_free =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(XMCAST_FREE_SEM)));
    if constexpr (XMCAST_ACTIVE) {
        noc_semaphore_set(x_ready, INVALID);
    }

    const uint32_t sem_data = static_cast<uint32_t>(get_semaphore(SEM_DATA));
    volatile tt_l1_ptr uint32_t* sem_data_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_data);
    uint32_t data_arrivals = 0;
    // The h-slice gather counter (scatter path, roots only). Monotone and cumulative.
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
    constexpr uint32_t WG_BLOCK_TILES = KR_PAD * HN_PAD;  // one gate weight K-block (num_k_blocks == 1)
    // The N-chunk the weight stream is published in. GU_CHUNKS == 1 restores the whole-block
    // push byte for byte (the chunk IS the block, and its row stride is HN_PAD again).
    constexpr uint32_t GU_CHUNK_W = HN_PAD / GU_CHUNKS;
    constexpr uint32_t WG_CHUNK_TILES = KR_PAD * GU_CHUNK_W;
    constexpr uint32_t REDUCE_SLOT_TILES = M_BLOCK * HN_PAD;  // one child's landing slot
    constexpr uint32_t X_ROW_BYTES = KR_PAD * BFP8_TILE;

    // One activation-row issue body for both the ordinary prologue and the cross-block prefetch.
    // On the bf16 path `dst` is one cb_x_in stick-row slot; on the tiled path it is the resident
    // cb_x_tiles row. Completion and CB publication stay with the caller.
    //
    // THE ONE PLACE x MEETS DRAM, which is why the fused extract is two additions and nothing else:
    // `row` is REGION-RELATIVE (callers clamp against M_T_MAX, so clamp-then-rebase composes) and
    // the base moves the read into this expert's slice.
    auto issue_x_row = [&](uint32_t row, uint32_t dst) {
        if constexpr (INPUT_FORMAT == 0) {
            for (uint32_t i = 0; i < TILE_H; ++i) {
                const uint32_t s = (i + my_col + my_row) % TILE_H;
                noc_async_read(
                    x_acc.get_noc_addr(x_stick_base + row * TILE_H + s, kstart * BF16_TILE_ROW_BYTES),
                    dst + s * X_SLICE,
                    kr_rows * BF16_TILE_ROW_BYTES);
            }
        } else {
            for (uint32_t i = 0; i < kr_rows; ++i) {
                noc_async_read(
                    x_acc.get_noc_addr(x_tile_base + row * EMB_T + kstart + i), dst + i * BFP8_TILE, BFP8_TILE);
            }
        }
    };

    // True means this core's next injector row, if any, has already landed.  Tiled input and the
    // ordinary depth-2 BF16 path also reserve cb_x_tiles here; the depth-1 BF16 pressure fallback
    // deliberately does not, because reserving the sole slot before phase 2 deadlocks with compute.
    bool x_prefetched = false;

    // `count == 0` -> m_blocks == 0 on every core: no CB traffic, no collective round, no
    // semaphore. Uniform across the grid, so it cannot hang.
    for (uint32_t block_idx = 0; block_idx < m_blocks; ++block_idx) {
        // The RUNTIME token tile-rows this block actually works on. Identical on every core (it is
        // a pure function of the same mailbox words), which is what keeps the three collectives'
        // round counts and landing addresses in lockstep across the grid.
        const uint32_t m_eff = moe_fused_swiglu::m_tiles_eff(m_t, block_idx, M_BLOCK, M_EFF_MIN);
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

        // Resident weights read DRAM on M-block 0 only: `cb_pop_front` advances a read pointer
        // without touching bytes, and each weight CB has a single producer, so a later block's slot
        // still holds block 0's data. Reserve/push/barrier/trip counts are unchanged.
        const bool read_wg = (block_idx == 0) || (W_RESIDENT == 0);
        const bool read_wd = (block_idx == 0) || (WD_RESIDENT == 0);

        // ---- Phase 1a: stage x, then multicast it along the grid row ----
        // cb_x_tiles is ONE slot, so its write pointer is the same L1 address on every core in the
        // row — what mcast_pipe requires of the landing address. W_gate is issued before the chain
        // and published after it.
        auto issue_wg_chunk = [&](uint32_t c) {
            cb_reserve_back(cb_w_gate, WG_CHUNK_TILES);
            MaybeDeviceZoneScope("reader_wg_issue");
            const uint32_t wg_wp = get_write_ptr(cb_w_gate);
            moe_fused_swiglu::read_weight_chunk<BRG>(
                wg_acc, read_wg, c, GU_CHUNK_W, kr_rows, kstart, hstart, hn_cols, HID_T, wg_wp, W_TILE);
        };
        // WHERE the tail chunks are issued is the whole result: issuing all GU_CHUNKS here
        // matters: the x staging prologue's read barrier is all-or-nothing, so issuing every chunk
        // up front drains the whole weight block before a single stick is tilized.
        // Chunk 0 here; chunks 1..N-1 after that barrier.

        const bool staged_early = x_prefetched;
        x_prefetched = false;
        constexpr bool PREFETCH_RESERVES_X_SLOT = (INPUT_FORMAT != 0) || (DEPTH_X > 1);
        if (!staged_early || !PREFETCH_RESERVES_X_SLOT) {
            cb_reserve_back(cb_x_tiles, x_slot_tiles);
        }
        const uint32_t x_base = get_write_ptr(cb_x_tiles);

        // ---- x staging PROLOGUE: land every tile-row THIS core injects, before the chain ----
        // Hoisted out of the multicast loop: staging is per-injector work with no cross-core
        // ordering, so doing it up front lets the chain run uninterrupted.
        {
            MaybeDeviceZoneScope("reader_xstage");
            // WHICH tile-row this column stages, and in WHAT ORDER its sticks are read. The walk
            // starts at `(my_col + my_row) % TILE_H` and wraps, so cores in a row are spread across
            // DRAM banks at any instant instead of contending for one.
            const uint32_t t_first = moe_fused_swiglu::inject_first(my_col);
            for (uint32_t t = t_first; t < m_eff; t += HGROUPS) {
                const uint32_t dst = x_base + t * X_ROW_BYTES;
                uint32_t row = block_idx * M_BLOCK + t;
                if (row >= M_T_MAX) {
                    row = M_T_MAX - 1;  // rows past the sized region are UNDEFINED; stay in bounds
                }
                if constexpr (INPUT_FORMAT == 0) {
                    // bf16 ROW_MAJOR: read this row-group's emb slice of 32 sticks. Compute tilizes
                    // it DIRECTLY into `dst`; cb_tilize_done is only a one-page completion channel, so
                    // the reader remains the sole owner of cb_x_tiles' push/write-pointer state.
                    if (!staged_early) {
                        {
                            MaybeDeviceZoneScope("reader_x_read");
                            cb_reserve_back(cb_x_in, TILE_H);
                            issue_x_row(row, get_write_ptr(cb_x_in));
                            noc_async_read_barrier();
                            cb_push_back(cb_x_in, TILE_H);
                        }
                    }

                    {
                        MaybeDeviceZoneScope("reader_x_tilize_wait");
                        cb_wait_front(cb_tilize_done, 1);
                        cb_pop_front(cb_tilize_done, 1);
                    }
                } else {
                    // bfp8_b TILE: the tiles land straight in the resident slot, no tilize.
                    if (!staged_early) {
                        issue_x_row(row, dst);
                        noc_async_read_barrier();
                    }
                }
            }
        }

        // This core's `x` is off DRAM. Release the writer's W_up stream (XPRIO). A plain
        // volatile store, not a NoC semaphore op: producer and consumer are two RISC-Vs on the SAME
        // core sharing one L1, and this word has exactly one writer. Monotone, so no reset.
        if constexpr (XPRIO) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_XSTAGED)) = block_idx + 1;
        }

        // The row-wide release targets the latency-critical one-block dispatch. Multi-block work
        // keeps the original pipeline: later X is prefetched, and perturbing its steady state costs
        // more than protecting the first block saves. With no row multicast there is no collective
        // rendezvous at all. Otherwise protect_x_stage releases W_gate inside round 0, only after
        // every core in the row has staged X, so an early core cannot starve a lagging core on NoC0.
        const bool protect_x_stage = (m_blocks == 1) && !staged_early;
        if constexpr (!XMCAST_ACTIVE) {
            issue_wg_chunk(0);
        } else if (!protect_x_stage) {
            issue_wg_chunk(0);
        }

        // ---- x multicast chain ----
        // m_eff rounds, not M_BLOCK: at count 128 that is half the handshake chain and half the
        // staged bytes. m_eff divides M_BLOCK, so the write pointer stays block-aligned and equal on
        // every core in the row. Full blocks publish per row; smaller ones keep the one-push handoff.
        {
            MaybeDeviceZoneScope("reader_xmcast");
            for (uint32_t t = 0; t < m_eff; ++t) {
                if constexpr (XMCAST_ACTIVE) {
                    // Round `t` carries tile-row `t`, injected by column `t % HGROUPS`. The lane
                    // value IS the sender's column, so the coord table needs no indirection.
                    const uint32_t round = t % HGROUPS;
                    if (round == my_col) {
                        noc_semaphore_wait(x_free, XMCAST_CONSUMERS);
                        noc_semaphore_set(x_free, 0);
                        if (t == 0 && protect_x_stage) {
                            // The normal round-0 free acknowledgements double as a row-wide
                            // X-staged barrier. Release W_gate uniformly, then use a second free
                            // acknowledgement to ensure every receiver consumed this phase before
                            // reusing x_ready for the payload-ready signal below.
                            constexpr uint32_t X_STAGED = 2;
                            issue_wg_chunk(0);
                            noc_semaphore_set(x_ready, X_STAGED);
                            noc_semaphore_set_multicast(
                                static_cast<uint32_t>(get_semaphore(XMCAST_READY_SEM)),
                                get_noc_multicast_addr(
                                    xbounds.sx,
                                    xbounds.sy,
                                    xbounds.ex,
                                    xbounds.ey,
                                    static_cast<uint32_t>(get_semaphore(XMCAST_READY_SEM))),
                                x_mcast_dests,
                                /*linked=*/false);
                            noc_async_writes_flushed();
                            noc_semaphore_set(x_ready, INVALID);
                            noc_semaphore_wait(x_free, XMCAST_CONSUMERS);
                            noc_semaphore_set(x_free, 0);
                        }
                        const uint32_t src = x_base + t * X_ROW_BYTES;
                        ncrisc_noc_fast_write_any_len<noc_mode>(
                            noc_index,
                            write_cmd_buf,
                            src,
                            get_noc_multicast_addr(xbounds.sx, xbounds.sy, xbounds.ex, xbounds.ey, src),
                            X_ROW_BYTES,
                            NOC_MULTICAST_WRITE_VC,
                            /*mcast=*/true,
                            /*linked=*/true,
                            x_mcast_dests,
                            /*multicast_path_reserve=*/true,
                            /*posted=*/false);
                        noc_semaphore_set(x_ready, VALID);
                        noc_semaphore_set_multicast(
                            static_cast<uint32_t>(get_semaphore(XMCAST_READY_SEM)),
                            get_noc_multicast_addr(
                                xbounds.sx,
                                xbounds.sy,
                                xbounds.ex,
                                xbounds.ey,
                                static_cast<uint32_t>(get_semaphore(XMCAST_READY_SEM))),
                            x_mcast_dests,
                            /*linked=*/false);
                        noc_async_writes_flushed();
                        // This core becomes a receiver in the next rotating round.
                        noc_semaphore_set(x_ready, INVALID);
                    } else {
                        const uint32_t sx = get_arg_val<uint32_t>(RT_XMCAST + 4 + 2 * round + 0);
                        const uint32_t sy = get_arg_val<uint32_t>(RT_XMCAST + 4 + 2 * round + 1);
                        noc_semaphore_inc(
                            get_noc_addr(sx, sy, static_cast<uint32_t>(get_semaphore(XMCAST_FREE_SEM))), 1);
                        if (t == 0 && protect_x_stage) {
                            constexpr uint32_t X_STAGED = 2;
                            noc_semaphore_wait(x_ready, X_STAGED);
                            noc_semaphore_set(x_ready, INVALID);
                            issue_wg_chunk(0);
                            noc_semaphore_inc(
                                get_noc_addr(sx, sy, static_cast<uint32_t>(get_semaphore(XMCAST_FREE_SEM))), 1);
                        }
                        noc_semaphore_wait(x_ready, VALID);
                        noc_semaphore_set(x_ready, INVALID);
                    }
                }
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

        // Phase 1b' — W_down for ALL WD_AHEAD phase-2 K-blocks, ISSUED as one batch, so the
        // reads land under the reduce rendezvous instead of in front of the round that needs them.
        constexpr uint32_t WD_BLOCK_TILES = HN_PAD * WD_EC_MAX;
        constexpr bool CAN_PREFETCH_X = HGROUPS >= M_BLOCK;
        const bool prefetch_next_x = kPrefetchNextX && CAN_PREFETCH_X && (block_idx + 1 < m_blocks);
        if (prefetch_next_x) {
            // Transaction id zero is the untagged stream and cannot be waited through the
            // scoped barrier on this architecture. Tag phase 2 before its first W_down issue.
            noc_async_read_set_trid(P2_READ_TRID);
        }
        auto issue_wd_batch = [&]() {
            const uint32_t nblocks = wd_mrow ? HGROUPS : WD_AHEAD;
            cb_reserve_back(cb_w_down, nblocks * WD_BLOCK_TILES);
            MaybeDeviceZoneScope("reader_wd_issue");
            const uint32_t write_ptr = get_write_ptr(cb_w_down);
            (void)write_ptr;
            for (uint32_t r = 0; r < nblocks; ++r) {
                const uint32_t hbase = moe_fused_swiglu::hidden_block_start(r, HID_T, HGROUPS, HN_PAD);
                const uint32_t block_hn_rows = moe_fused_swiglu::hidden_block_rows(r, HID_T, HGROUPS, HN_PAD);
                // The writer takes the TAIL rows on NOC_1; this is the head.
                if (read_wd) {
                    moe_fused_swiglu::read_wd_rows<BRD>(
                        wd_acc,
                        hbase,
                        0,
                        block_hn_rows - wd_rows_writer(block_hn_rows),
                        wd_jstart,
                        wd_out_width,
                        WD_EC_MAX,
                        EMB_T,
                        WD_PACKED ? wd_base + hbase * WD_EC_MAX * W_TILE : write_ptr + r * WD_BLOCK_TILES * W_TILE,
                        W_TILE);
                }
            }
        };
        issue_wd_batch();

        // Start block_idx+1's activation read before block_idx's reduce + phase 2. At the supported
        // grids HGROUPS >= M_BLOCK, so each core injects at most one row and the existing one-row
        // cb_x_in is sufficient. Smaller grids retain the ordinary next-block prologue.
        bool prefetch_has_local_row = false;
        if (prefetch_next_x) {
            const uint32_t next_m_eff = moe_fused_swiglu::m_tiles_eff(m_t, block_idx + 1, M_BLOCK, M_EFF_MIN);
            uint32_t next_x_base = 0;
            if constexpr (PREFETCH_RESERVES_X_SLOT) {
                cb_reserve_back(cb_x_tiles, next_m_eff * KR_PAD);
                next_x_base = get_write_ptr(cb_x_tiles);
            }
            const uint32_t t = moe_fused_swiglu::inject_first(my_col);
            if (t < next_m_eff) {
                prefetch_has_local_row = true;
                uint32_t row = (block_idx + 1) * M_BLOCK + t;
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
        // SEM_GO is the invite: the flow control stopping a contributor from overwriting a landing
        // slot compute has not consumed. The slice comes from the one shared plan in common.hpp.
        const uint32_t slice_worker_count = moe_fused_swiglu::slice_workers(h_block_tiles, KGROUPS);
        const uint32_t slice_tiles_each =
            (slice_worker_count != 0) ? (h_block_tiles / slice_worker_count) : 0;           // uniform slice size
        const uint32_t slice_tiles = (my_row < slice_worker_count) ? slice_tiles_each : 0;  // 0 = an idle core
        MaybeDeviceZoneScope("reader_reduce");
        // Reserve the landing CBs WHOLE first, THEN invite the whole column, then wait for every
        // contributor and push WHOLE. This is also cb_h_local's flow control, transitively: my
        // invite for block_idx+1 is issued after my phase 2 of block_idx has read it, and no worker's
        // h-slice send for block_idx+1 can precede this invite. So the h landing needs no second handshake.
        {
            MaybeDeviceZoneScope("reader_reduce_reserve");
            if (slice_tiles) {
                cb_reserve_back(cb_gather_gate, GATHER_PAGES);
                cb_reserve_back(cb_gather_up, GATHER_PAGES);
            }
        }
        // cb_gather_gate aliases cb_h_slice and cb_out_tiles at bfp8 output. Reserving touches only
        // this view's FIFO state, but the invite authorises PEERS to write the shared physical SRAM
        // — so block_idx must wait for this core's writer to drain block_idx-1's output DMA.
        {
            MaybeDeviceZoneScope("reader_reduce_phase_wait");
            if constexpr (PHASE_CB_ALIAS) {
                if (block_idx != 0) {
                    moe_fused_swiglu::sem_wait_min(SEM_PHASE_FREE, block_idx);
                }
            }
        }
        const uint32_t sem_go = static_cast<uint32_t>(get_semaphore(SEM_GO));
        {
            MaybeDeviceZoneScope("reader_reduce_invite");
            for (uint32_t i = 0; i < KGROUPS; ++i) {
                const uint32_t p = i;
                const uint32_t px = get_arg_val<uint32_t>(RT_PEERS + 2 * p + 0);
                const uint32_t py = get_arg_val<uint32_t>(RT_PEERS + 2 * p + 1);
                noc_semaphore_inc(get_noc_addr(px, py, sem_go), 1);
            }
            noc_async_atomic_barrier();
        }
        {
            {
                MaybeDeviceZoneScope("reader_reduce_up_wait");
                cb_wait_front(cb_up_acc, h_block_tiles);
            }
            // The UP half of the column all-to-all, on NOC_0; the writer carries the GATE half.
            // Wait for the WHOLE column's invites first, exactly as the writer does: every core
            // invites once per peer per M-block, so (block_idx+1)*KGROUPS is the exact total.
            {
                MaybeDeviceZoneScope("reader_reduce_invite_wait");
                moe_fused_swiglu::sem_wait_min(SEM_GO, (block_idx + 1) * KGROUPS);
            }
            if constexpr (SCATTER_ONE_SIGNAL) {
                {
                    MaybeDeviceZoneScope("reader_reduce_up_payload");
                    moe_fused_swiglu::scatter_payload(
                        RT_PEERS, cb_up_acc, cb_gather_up, slice_worker_count, slice_tiles_each, my_row, BFP8_TILE);
                }
                // The payload barrier in scatter_payload is the data-before-publish proof.  The
                // writer invalidates while polling this monotone mailbox word, then signals the
                // destination only after its independent gate payload has landed too.
                asm volatile("fence" ::: "memory");
                mailbox_words[moe_fused_swiglu::MBOX_UP_SCATTER_DONE] = block_idx + 1;
            } else {
                moe_fused_swiglu::scatter_leg(
                    RT_PEERS,
                    cb_up_acc,
                    cb_gather_up,
                    SEM_DATA,
                    slice_worker_count,
                    slice_tiles_each,
                    my_row,
                    BFP8_TILE);
            }
            cb_pop_front(cb_up_acc, h_block_tiles);
        }
        if (slice_tiles) {
            // One signal per payload by default; SCATTER_ONE_SIGNAL keeps both payloads concurrent
            // but has the source writer signal once, after both have landed.
            data_arrivals += (SCATTER_ONE_SIGNAL ? 1 : 2) * KGROUPS;
            {
                MaybeDeviceZoneScope("reader_reduce_data_wait");
                noc_semaphore_wait_min(sem_data_ptr, data_arrivals);
            }
            cb_push_back(cb_gather_gate, GATHER_PAGES);
            cb_push_back(cb_gather_up, GATHER_PAGES);
        }

        // ---- Phase 2: broadcast h and consume W_down ----
        // Full M-blocks broadcast whole H per M row against the resident W_down shard. Ragged blocks
        // keep the hidden-slice rounds and their deferred barrier, so each W_down read lands under
        // the next multicast; `wd_pending` carries the issued-but-unpublished block between rounds.
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
        // On the scatter path this column's h block is ASSEMBLED IN cb_h_local BY THE
        // WORKERS' NoC WRITES, not packed by compute, so the root's handshake is the SEM_HSLICE
        // counter rather than a CB front. `slice_worker_count` slices land per M-block; the counter is monotone and
        // cumulative, like every other semaphore in this op.
        if (wd_mrow) {
            if (is_row_agg) {
                h_arrivals += HGROUPS;
            }
        } else if (is_root) {
            h_arrivals += slice_worker_count;
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
                // Ordinary mode broadcasts all eight token rows across 88 cores. Large-M grouped
                // mode runs two concurrent four-round schedules over rows 0..3 and 4..7; each
                // receiver only ingests its group's rows and each sender waits for 44, not 88,
                // window acks. W_down's complete resident shard is already published.
                const uint32_t round_count = wd_mgroup ? MGROUP_ROWS : KGROUPS;
                const uint32_t round_base = wd_mgroup ? (my_row / MGROUP_ROWS) * MGROUP_ROWS : 0;
                uint32_t next_ack = 0;
                for (uint32_t lr = 0; lr < round_count; ++lr) {
                    const uint32_t r = round_base + lr;
                    uint32_t ahead = HACK_AHEAD;
                    if (ahead > DEPTH_H) {
                        ahead = DEPTH_H;
                    }
                    if (ahead > round_count - lr) {
                        ahead = round_count - lr;
                    }
                    cb_reserve_back(cb_h, ahead * HROW_T);
                    const uint32_t hdst = get_write_ptr(cb_h);
                    while (next_ack < lr + ahead && next_ack < round_count) {
                        const uint32_t gr = round_base + next_ack;
                        const uint32_t sidx = gr * HGROUPS + gr;
                        const uint32_t svx = get_arg_val<uint32_t>(RT_HMCAST + 4 + 2 * sidx + 0);
                        const uint32_t svy = get_arg_val<uint32_t>(RT_HMCAST + 4 + 2 * sidx + 1);
                        noc_semaphore_inc(get_noc_addr(svx, svy, static_cast<uint32_t>(get_semaphore(SEM_H_FREE))), 1);
                        ++next_ack;
                    }

                    const bool i_send = is_row_agg && (my_row == r);
                    bool writer_owns_send = !wd_mgroup && h_round_on_writer(r);
                    const uint32_t slot =
                        wd_mgroup ? ((block_idx * MGROUP_ROWS + lr) % DEPTH_H) : ((block_idx * KGROUPS + r) % DEPTH_H);
                    if (i_send && !writer_owns_send) {
                        noc_semaphore_wait_min(sem_h_ptr, h_arrivals);
                        noc_async_read(get_noc_addr(get_write_ptr(cb_h_local)), hdst, HROW_T * H_TILE);
                        phase2_read_barrier();
                        if constexpr (HMCAST_ACTIVE) {
                            h_free_expected += wd_mgroup ? MGROUP_CORES : NUM_CORES;
                            noc_semaphore_wait_min(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                    static_cast<uint32_t>(get_semaphore(SEM_H_FREE))),
                                h_free_expected);
                            h_slot_send_posted(slot, hdst, HROW_T * H_TILE, wd_mgroup);
                        }
                    } else if (i_send) {
                        // This diagonal core is excluded from its writer's multicast, just like a
                        // reader-owned sender.  The writer publishes only after its NoC1 self-copy
                        // and linked payload+flag chain have flushed, so this is both local-data
                        // readiness and source-slot reuse safety.
                        while (mailbox_words[moe_fused_swiglu::MBOX_HSEND_DONE] < block_idx + 1) {
                            invalidate_l1_cache();
                        }
                    } else {
                        if constexpr (HMCAST_ACTIVE) {
                            volatile tt_l1_ptr uint32_t* hf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                static_cast<uint32_t>(get_semaphore(SEM_H_RDY_BASE + slot)));
                            MaybeDeviceZoneScope("p2_hwait");
                            noc_semaphore_wait(hf, VALID);
                            noc_semaphore_set(hf, INVALID);
                        }
                    }
                    cb_push_back(cb_h, HROW_T);
                }
                if (!wd_mgroup) {
                    // Eight 64-tile pushes advance a 3x64-tile CB by two slots. Publish one
                    // payload-free slot so an ordinary dispatch can switch to a smaller tail.
                    cb_reserve_back(cb_h, HROW_T);
                    cb_push_back(cb_h, HROW_T);
                }
            } else {
                bool wd_pending = false;
                // HACK_AHEAD — rounds' senders acked in one go. The all-gather's per-round rendezvous
                // is a fixed cost that exceeds the work it guards, so amortizing it over several
                // rounds is the lever; legal only because the VALID cells are per-slot. Clamped here
                // because m_eff is runtime.
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
                        // VALID, so the slot can be read before the write lands.
                        // cb_h_local needs no CB front — the workers' NoC writes assemble it.
                        noc_semaphore_wait_min(sem_h_ptr, h_arrivals);
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
                        // PER-SLOT FLAGS. Linked data+signal multicast, so still NO acked write
                        // barrier, and the VALID cell is this SLOT's — round r+1's sender is not held
                        // behind every core clearing round r's. The ack accounting is the MONOTONE
                        // `h_free_expected` counter, because HACK_AHEAD deliberately breaks the
                        // round-to-round chain a reset-based handshake would need.
                        if constexpr (HMCAST_ACTIVE) {
                            h_free_expected += NUM_CORES;
                            noc_semaphore_wait_min(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                    static_cast<uint32_t>(get_semaphore(SEM_H_FREE))),
                                h_free_expected);
                            // `src == dst`, so exclude-source: a src != dst send is a LOOPBACK
                            // multicast whose flag reset races this core's own in-flight VALID.
                            h_slot_send_posted((block_idx * HGROUPS + r) % DEPTH_H, hdst, h_block_tiles * H_TILE);
                        }
                    } else {
                        // Every core but this round's sender drains AFTER the multicast, so the W_down read
                        // had this whole round's grid-wide broadcast to land under — that is the deferral.
                        // The whole of `ReceiverPipe::receive` for a Flag signal with PRE_HANDSHAKE
                        // off: wait for THIS slot's VALID, then put it back. Raw rather than a per-slot
                        // ReceiverPipe because that class's ctor sets the cell INVALID and would clobber
                        // a VALID a sender running ahead had already broadcast.
                        if constexpr (HMCAST_ACTIVE) {
                            volatile tt_l1_ptr uint32_t* hf =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(
                                    get_semaphore(SEM_H_RDY_BASE + ((block_idx * HGROUPS + r) % DEPTH_H))));
                            MaybeDeviceZoneScope("p2_hwait");
                            noc_semaphore_wait(hf, VALID);
                            noc_semaphore_set(hf, INVALID);
                        }
                        if (wd_pending) {
                            // The REAL per-round W_down wait. `reader_wd_wait` covers only the
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
                        const uint32_t hbase = moe_fused_swiglu::hidden_block_start(pre, HID_T, HGROUPS, HN_PAD);
                        const uint32_t block_hn_rows = moe_fused_swiglu::hidden_block_rows(pre, HID_T, HGROUPS, HN_PAD);
                        cb_reserve_back(cb_w_down, WD_BLOCK_TILES);
                        const uint32_t write_ptr = get_write_ptr(cb_w_down);
                        // The writer already has the tail rows in flight on NOC_1 (it issued every
                        // K-block as one batch back in phase 1), so this reads only the head.
                        if (read_wd) {
                            moe_fused_swiglu::read_wd_rows<BRD>(
                                wd_acc,
                                hbase,
                                0,
                                block_hn_rows - wd_rows_writer(block_hn_rows),
                                wd_jstart,
                                wd_out_width,
                                WD_EC_MAX,
                                EMB_T,
                                WD_PACKED ? wd_base + hbase * WD_EC_MAX * W_TILE : write_ptr,
                                W_TILE);
                        }
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
