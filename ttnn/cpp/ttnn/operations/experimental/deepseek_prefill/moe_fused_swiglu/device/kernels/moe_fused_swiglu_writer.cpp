// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — WRITER (NoC1).
//
// Per M-block, in order:
//   1. drain the PREVIOUS block's output write-back (the deferred write barrier);
//   2. the W_up weight stream — the NoC1 twin of the reader's W_gate stream, so a phase with two
//      independent weight streams uses both data-movement RISC-Vs and both NoCs;
//   3. this core's share of the phase-2 W_down stream, published per K-block by transaction id;
//   4. the GATE half of the reduce-scatter's column all-to-all (the reader carries the UP half),
//      then this core's finished `h` slice straight into the column root's cb_h_local — the
//      gather IS the assembly;
//   5. issue the output write-back, coalesced over the emb axis and clamped to tile-rows below
//      ceil_tile(count) so rows past the real token count are never touched.
//
// Raw-dataflow deviations are the reader's, for the same reasons. The transport vocabulary the two
// kernels share lives in moe_fused_swiglu_dataflow.hpp.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/debug/assert.h"

#include "tt_metal/tools/profiler/kernel_profiler.hpp"

#include "moe_fused_swiglu_dataflow.hpp"  // the transport vocabulary shared with the reader
#include "moe_fused_swiglu_common.hpp"    // the ONE definition of the mailbox word layout
#include "moe_fused_swiglu_ct_args.hpp"   // the ONE definition of the compile-time arg order

#define MaybeDeviceZoneScope(name) DeviceZoneScopedN(name)

MOE_DECLARE_CT_ENUM(MOE_WRITER_CT_ARGS);

constexpr uint32_t EMB_T = CT(EMB_T);
constexpr uint32_t HID_T = CT(HID_T);
constexpr uint32_t KR_PAD = CT(KR_PAD);
constexpr uint32_t HN_PAD = CT(HN_PAD);
constexpr uint32_t EC_MAX = CT(EC_MAX);  // phase-2 N stride (uniform CB increment)
constexpr uint32_t WD_EC_MAX = CT(WD_EC_MAX);
constexpr uint32_t EC_GROUP_MAX = CT(EC_GROUP_MAX);
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
constexpr uint32_t SEM_PHASE_FREE = CT(SEM_PHASE_FREE);
constexpr uint32_t SEM_HROW_FREE = CT(SEM_HROW_FREE);
constexpr uint32_t PHASE_CB_ALIAS = CT(PHASE_CB_ALIAS);

constexpr uint32_t W_TILE = CT(W_TILE_BYTES);  // weight tile stride: bfp4 576, bfp8 1088, bf16 2048
constexpr uint32_t BFP8_TILE = CT(BFP8_TILE);
// The OUTPUT tile size, which is NOT BFP8_TILE once the caller passes `dtype=`: a bf16 output
// tile is 2048 B, and striding the write-back by 1088 would emit partial pages from wrong offsets.
constexpr uint32_t OUT_TILE = CT(OUT_TILE_BYTES);
constexpr uint32_t H_TILE = BFP8_TILE;  // h is bfp8; see the reader
constexpr uint32_t MAILBOX_MAGIC = CT(MAILBOX_MAGIC);
constexpr uint32_t M_EFF_MIN = CT(M_EFF_MIN);
// Cross-M-block weight residency, the NoC1 half: W_up's read carries no M-block index, so every
// block after the first re-reads bytes still resident in cb_w_up's slot.
constexpr uint32_t W_RESIDENT = CT(W_RESIDENT);
constexpr uint32_t WD_RESIDENT = CT(WD_RESIDENT);
constexpr bool WD_PACKED = WD_RESIDENT && moe_fused_swiglu::hidden_blocks_are_balanced(HID_T, HGROUPS, HN_PAD);
constexpr uint32_t GU_CHUNKS = CT(GU_CHUNKS);
constexpr uint32_t XPRIO = CT(XPRIO);
constexpr uint32_t WD_MROW_ROUNDS = CT(WD_MROW_ROUNDS);
constexpr uint32_t WD_MGROUPS = CT(WD_MGROUPS);
constexpr uint32_t WD_MGROUP_MIN_BLOCKS = CT(WD_MGROUP_MIN_BLOCKS);
constexpr uint32_t MGROUP_ROWS = M_BLOCK / 2;
constexpr uint32_t DEPTH_H = CT(DEPTH_H);
constexpr uint32_t H_ROUND_NOC1_MASK = CT(H_ROUND_NOC1_MASK);
constexpr uint32_t SCATTER_ONE_SIGNAL = CT(SCATTER_ONE_SIGNAL);
constexpr uint32_t WD_SPLIT = CT(WD_SPLIT);
constexpr uint32_t GATHER_PAGES = CT(GATHER_PAGES);
constexpr uint32_t PHASE_ALIAS_PAGES = CT(PHASE_ALIAS_PAGES);
constexpr uint32_t WG_SHARD_W = CT(WG_SHARD_W);
constexpr uint32_t WD_SHARD_W = CT(WD_SHARD_W);
// DIRECT_WRITE: the output tensor is a SHARED destination buffer and this expert's rows go in at
// its region base instead of row 0 — what ttnn::insert used to do, with no temp buffer and no
// second DRAM round-trip. The base itself comes from the mailbox (the reader fetched it in phase
// 0), so this kernel needs no `start` accessor and no DRAM read of its own.
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t DIRECT_WRITE = CT(DIRECT_WRITE);
// Tile-rows of the DESTINATION tensor. Equals this expert's region height in the ordinary mode and
// the whole shared buffer's height under DIRECT_WRITE; either way it BOUNDS every write.
constexpr uint32_t OUT_M_T = CT(OUT_M_T);

constexpr uint32_t cb_w_up = CT(CB_W_UP);
constexpr uint32_t cb_w_down = CT(CB_W_DOWN);
constexpr uint32_t cb_out_tiles = CT(CB_OUT_TILES);
constexpr uint32_t cb_gate_acc = CT(CB_GATE_ACC);
constexpr uint32_t cb_up_acc = CT(CB_UP_ACC);
constexpr uint32_t cb_gather_gate = CT(CB_GATHER_GATE);
constexpr uint32_t cb_gather_up = CT(CB_GATHER_UP);
constexpr uint32_t cb_h_slice = CT(CB_H_SLICE);
constexpr uint32_t cb_h_local = CT(CB_H_LOCAL);
constexpr uint32_t cb_h = CT(CB_H);
constexpr uint32_t cb_mailbox_writer = CT(CB_MAILBOX_WRITER);

constexpr uint32_t RT_PEERS = 17;  // KGROUPS (vx, vy) pairs — the whole column, in row order
constexpr uint32_t RT_HRECT = RT_PEERS + 2 * KGROUPS;
constexpr bool kHMcastPosted = (H_MCAST_POSTED != 0);

inline bool h_round_on_writer(uint32_t r) { return ((H_ROUND_NOC1_MASK >> r) & 1u) != 0; }

// A complete writer-owned round.  Payload, linked flag, flush and the rotating-sender local reset
// all stay on this RISC/NoC, which is the ownership boundary the earlier byte-wise split violated.
inline void h_slot_send_posted_noc1(uint32_t slot, uint32_t l1, uint32_t size) {
    const auto hrect = moe_fused_swiglu::McastRect<noc_index>(
        get_arg_val<uint32_t>(RT_HRECT + 0),
        get_arg_val<uint32_t>(RT_HRECT + 1),
        get_arg_val<uint32_t>(RT_HRECT + 2),
        get_arg_val<uint32_t>(RT_HRECT + 3));
    const auto& rb = hrect.bounds();
    constexpr uint32_t ndest = NUM_CORES - 1;
    const uint32_t hf_addr = static_cast<uint32_t>(get_semaphore(SEM_H_RDY_BASE + slot));
    volatile tt_l1_ptr uint32_t* hf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hf_addr);

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
    noc_semaphore_set(hf, VALID);
    noc_semaphore_set_multicast(
        hf_addr, get_noc_multicast_addr(rb.sx, rb.sy, rb.ex, rb.ey, hf_addr), ndest, /*linked=*/false);
    noc_async_writes_flushed();
    noc_semaphore_set(hf, INVALID);
}

// The accessor block follows the scalar block; CT_COUNT is its length.
constexpr auto wu_args = TensorAccessorArgs<CT_COUNT>();
constexpr auto out_args = TensorAccessorArgs<wu_args.next_compile_time_args_offset()>();
constexpr auto wd_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

// The W_down NoC split: WD_SPLIT eighths of every phase-2 K-block's hidden rows are read HERE on
// NOC_1 instead of on the reader's NOC_0, and published per K-block by transaction id.
constexpr uint32_t WD_BLOCK_TILES = HN_PAD * WD_EC_MAX;  // the reader's twin — one phase-2 K-block
// The W_down stream takes its OWN tensor's DRAM ND shard width, exactly as the reader's `BRD` does.
using BRD = moe_fused_swiglu::WeightRuns<WD_SHARD_W>;

// Bank-run coalescing (see moe_fused_swiglu_bank_runs.hpp): ONE definition, bound here to this
// kernel's compile-time knobs — identical to the reader's binding, which is the point.
using BR = moe_fused_swiglu::WeightRuns<>;
// The W_up stream takes its tensor's DRAM ND shard width (0 = interleaved, byte-identical to the
// pre-WSHARD path). The OUTPUT write-back keeps `BR`: the output is always DRAM interleaved.
using BRG = moe_fused_swiglu::WeightRuns<WG_SHARD_W>;

// PER-STAGE ZONES — PERMANENT, always compiled, free with the profiler off. 5 records per
// M-block: out_drain, wup, wd_issue, scatter, hslice, out_issue. Names are stable across rounds so
// old and new numbers stay comparable; give any new fast path its own zone.

void kernel_main() {
    (void)get_arg_val<uint32_t>(0);  // retained runtime slot for cache-compatible argument layout
    const uint32_t w_up_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t wd_addr = get_arg_val<uint32_t>(3);
    const uint32_t kr = get_arg_val<uint32_t>(4);
    const uint32_t kstart = get_arg_val<uint32_t>(5);
    const uint32_t hstart = get_arg_val<uint32_t>(6);
    const uint32_t hn = get_arg_val<uint32_t>(7);
    const uint32_t ec = get_arg_val<uint32_t>(8);
    const uint32_t jstart = get_arg_val<uint32_t>(9);
    const uint32_t ec_group = get_arg_val<uint32_t>(10);
    const uint32_t jstart_group = get_arg_val<uint32_t>(11);
    const uint32_t my_col = get_arg_val<uint32_t>(12);
    // My row in the grid column. It IS my contributor slot in every peer's landing CB and it IS the
    // index of the slice I own, so the scatter needs no host-side plan table.
    const uint32_t my_row = get_arg_val<uint32_t>(13);
    // The row of THIS column's reduce root (`x % KGROUPS`) — the core the finished h slices are
    // gathered into, since it injects this column's h into the phase-2 all-gather.
    const uint32_t root_row = get_arg_val<uint32_t>(14);
    const bool is_root = (my_row == root_row);
    const bool is_row_agg = (my_col == my_row);
    const uint32_t row_agg_vx = get_arg_val<uint32_t>(15);
    const uint32_t row_agg_vy = get_arg_val<uint32_t>(16);
    const auto wu_acc = TensorAccessor(wu_args, w_up_addr, W_TILE);
    const auto out_acc = TensorAccessor(out_args, out_addr, OUT_TILE);
    const auto wd_acc = TensorAccessor(wd_args, wd_addr, W_TILE);
    // THE ADDRESS DERIVATION, and why it needs no CB state. This RISC-V never pushes cb_w_down
    // (the reader is its single producer), so its local `cb_interface` copy never advances and
    // `get_write_ptr` is the CB BASE for the whole kernel. Residency forces the capacity to
    // exactly HGROUPS K-blocks, so K-block r lives at `base + r * WD_BLOCK_TILES * W_TILE`.
    const uint32_t wd_base = get_write_ptr(cb_w_down);

    // The reader owns the device-resident count read and publishes it to the
    // program-local aliased mailbox CB. The FIFO event makes the raw payload
    // read safe even when L1 contains bytes from an earlier program.
    cb_wait_front(cb_mailbox_writer, 1);
    const uint32_t mailbox_addr = get_read_ptr(cb_mailbox_writer);
    const auto mb = moe_fused_swiglu::mailbox_wait(mailbox_addr, MAILBOX_MAGIC, [] { invalidate_l1_cache(); });
    cb_pop_front(cb_mailbox_writer, 1);
    volatile tt_l1_ptr uint32_t* mailbox_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_addr);
    const uint32_t m_t = mb.m_t;
    const uint32_t m_blocks = mb.m_blocks;
    const bool wd_mgroup = WD_MGROUPS && (m_blocks >= WD_MGROUP_MIN_BLOCKS) && (m_t != 0) && ((m_t % M_BLOCK) == 0);
    const uint32_t wd_ec = wd_mgroup ? ec_group : ec;
    const uint32_t wd_jstart = wd_mgroup ? jstart_group : jstart;
    // The destination rebase, in TILE-ROWS. `mb.start_row` is a token row the reader fetched from
    // start[global_expert_id]; flooring by TILE_HEIGHT mirrors ttnn::insert's writer exactly, and
    // the reader ASSERTs the alignment that makes the floor lossless. It MUST agree with the
    // reader's x rebase or this expert would read one region and write another.
    const uint32_t out_row_base = DIRECT_WRITE ? mb.start_row / TILE_HEIGHT : 0;

    volatile tt_l1_ptr uint32_t* sem_go_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_GO)));
    volatile tt_l1_ptr uint32_t* phase_free_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_PHASE_FREE)));
    uint32_t invites = 0;
    constexpr uint32_t WU_BLOCK_TILES = KR_PAD * HN_PAD;
    // The N-chunk width of the gate/up weight stream. 1 is the whole block.
    constexpr uint32_t GU_CHUNK_W = HN_PAD / GU_CHUNKS;
    constexpr uint32_t WU_CHUNK_TILES = KR_PAD * GU_CHUNK_W;
    constexpr uint32_t SLOT_TILES = M_BLOCK * HN_PAD;  // one child's landing slot in the parent

    // The output block written but not yet barriered/popped — see the DEFERRED WRITE BARRIER below.
    uint32_t out_pending = 0;

    for (uint32_t b = 0; b < m_blocks; ++b) {
        // The RUNTIME token tile-rows this block works on — the SAME number the reader uses for its
        // multicast rounds and compute uses for its matmul shape (moe_fused_swiglu_common.hpp).
        const uint32_t m_eff = moe_fused_swiglu::m_tiles_eff(m_t, b, M_BLOCK, M_EFF_MIN);
        const bool wd_mrow = WD_MROW_ROUNDS && (m_eff == M_BLOCK);
        const uint32_t gu_block_tiles = m_eff * HN_PAD;
        const uint32_t out_rows = wd_mgroup ? MGROUP_ROWS : m_eff;
        const uint32_t out_ec_max = wd_mgroup ? EC_GROUP_MAX : EC_MAX;
        const uint32_t out_ec = wd_mgroup ? ec_group : ec;
        const uint32_t out_jstart = wd_mgroup ? jstart_group : jstart;
        const uint32_t out_row_in_block = wd_mgroup ? (my_row / MGROUP_ROWS) * MGROUP_ROWS : 0;
        const uint32_t out_block_tiles = out_rows * out_ec_max;

        // DEFERRED WRITE BARRIER, the twin of the reader's deferred READ barrier. The previous
        // M-block's write-back is drained HERE, not where it was issued: barriering at the issue
        // site made the last stage of block b pay full DRAM write latency with nothing else in
        // flight, sitting exactly on the multi-M-block critical path. DEPTH_OUT >= 2 makes the
        // extra outstanding block legal.
        if (out_pending) {
            MaybeDeviceZoneScope("writer_out_drain");
            noc_async_write_barrier();
            cb_pop_front(cb_out_tiles, out_pending);
            out_pending = 0;
            // BFP8 phase alias: only after the DMA no longer reads cb_out_tiles may this core's
            // reader invite peers to overwrite the same SRAM through cb_gather_gate. The value is
            // the next block index and is monotone, like every other same-core publication here.
            if constexpr (PHASE_CB_ALIAS) {
                *phase_free_ptr = b;
            }
        }

        // ---- W_up: NoC1 half of the gate/up weight stream, same bank-run coalescing ----
        //
        // Published in the same `GU_CHUNKS` N-chunks as the reader's W_gate twin, for the
        // same reason: the up matmul starts on chunk 0 while chunk 1 is still in DRAM. See the
        // reader's comment for why N (independent chunks, no extra accumulating pack) and not K.
        {
            MaybeDeviceZoneScope("writer_wup");
            // Activation-first: hold this NoC1 weight stream until this core's reader has
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
                // Residency: M-block 0 only, because the read carries no `b`.
                moe_fused_swiglu::read_weight_chunk<BRG>(
                    wu_acc, (b == 0) || (W_RESIDENT == 0), c, GU_CHUNK_W, kr, kstart, hstart, hn, HID_T, wp, W_TILE);
                noc_async_read_barrier();
                cb_push_back(cb_w_up, WU_CHUNK_TILES);
            }
        }

        // ---- MY SHARE OF THE PHASE-2 W_down STREAM, on NOC_1 ----
        // WD_SPLIT eighths of every K-block's hidden rows, read here so they do not compete with
        // the h all-gather for the reader's NOC_0. All HGROUPS blocks go out as ONE batch: with
        // residency every W_down read happens at b == 0, where all slots are free from kernel
        // start. This RISC-V takes the tail rows, a contiguous run.
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
                const uint32_t hbase = moe_fused_swiglu::hidden_block_start(r, HID_T, HGROUPS, HN_PAD);
                const uint32_t hn_r = moe_fused_swiglu::hidden_block_rows(r, HID_T, HGROUPS, HN_PAD);
                const uint32_t k_w = (hn_r * WD_SPLIT) / 8;  // the TAIL rows, read HERE
                // ONE TRANSACTION ID PER K-BLOCK. Every block still goes to DRAM at once (that
                // concurrency is the point of the batch); tagging lets the drain below release
                // block r when IT lands rather than when the last one does.
                noc_async_read_set_trid(r + 1);
                moe_fused_swiglu::read_wd_rows<BRD>(
                    wd_acc,
                    hbase,
                    hn_r - k_w,
                    hn_r,
                    wd_jstart,
                    wd_ec,
                    WD_EC_MAX,
                    EMB_T,
                    wd_base + (WD_PACKED ? hbase * WD_EC_MAX : r * WD_BLOCK_TILES) * W_TILE,
                    W_TILE);
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
            // KGROUPS invites per M-block is the generalisation of the tree's one-parent SEM_GO:
            // every peer's reader reserves its landing CBs and THEN invites the whole column, so
            // this wait is what stops block b+1's contribution from overwriting a landing slot
            // compute has not consumed yet. MONOTONE, never reset — the running total only grows.
            noc_semaphore_wait_min(sem_go_ptr, invites + KGROUPS);
            invites += KGROUPS;
            cb_wait_front(cb_gate_acc, gu_block_tiles);
            uint32_t gate_dst = get_write_ptr(cb_gather_gate);
            if constexpr (PHASE_CB_ALIAS) {
                // The three BFP8 phase views have the physical LCM capacity, which can exceed
                // GATHER_PAGES (N=1024: 48 vs 24). The reader advances the logical gather cursor;
                // this writer's local CB-interface cursor is not a valid remote-address proxy in
                // that case. Derive the identical physical slot directly from block index.
                const auto& iface = get_local_cb_interface(cb_gather_gate);
                const uint32_t base = iface.fifo_limit - iface.fifo_size;
                gate_dst = base + ((b * GATHER_PAGES) % PHASE_ALIAS_PAGES) * H_TILE;
            }
            // The GATE half of the column all-to-all, on NOC_1. The reader carries the UP half on
            // NOC_0: split by PAYLOAD, not destination, so each RISC-V owns one
            // accumulator CB outright — two RISC-Vs popping one CB corrupt its shared `tiles_acked`
            // word the way two pushers corrupt `tiles_received`.
            if constexpr (SCATTER_ONE_SIGNAL) {
                moe_fused_swiglu::scatter_payload_to(RT_PEERS, cb_gate_acc, gate_dst, sl_w, sl_a, my_row, H_TILE);
                while (mailbox_words[moe_fused_swiglu::MBOX_UP_SCATTER_DONE] < b + 1) {
                    invalidate_l1_cache();
                }
                // Both NoC payload barriers have now completed. One signal per source/destination
                // pair proves both gate and up are resident in the target landing buffers.
                moe_fused_swiglu::scatter_signal(RT_PEERS, SEM_DATA, sl_w);
            } else {
                moe_fused_swiglu::scatter_payload_to(RT_PEERS, cb_gate_acc, gate_dst, sl_w, sl_a, my_row, H_TILE);
                moe_fused_swiglu::scatter_signal(RT_PEERS, SEM_DATA, sl_w);
            }
            cb_pop_front(cb_gate_acc, gu_block_tiles);
        }
        // ---- my finished h slice, straight into the ROOT's cb_h_local at its tile offset ----
        // The gather IS the assembly. cb_h_local is never pushed or popped, so its write pointer
        // is the CB base on every core — which is what lets this core use its OWN pointer as the
        // root's. Flow control is the invite above, transitively: my send is downstream of the
        // gather, which is downstream of the root's invite for this block.
        if (my_row < sl_w) {
            MaybeDeviceZoneScope("writer_hslice");
            cb_wait_front(cb_h_slice, sl_a);
            uint32_t rvx;
            uint32_t rvy;
            uint32_t dst;
            uint32_t bytes;
            if (wd_mrow) {
                // The full-M reduce gives row r exactly one HN_PAD-wide token tile-row in every
                // hidden column.  Gather those eleven adjacent fragments horizontally onto the
                // diagonal row aggregator, producing one contiguous HID_T-wide W_down operand.
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_HROW_FREE))),
                    b + 1);
                rvx = row_agg_vx;
                rvy = row_agg_vy;
                dst = get_write_ptr(cb_h_local) + hstart * H_TILE;
                bytes = hn * H_TILE;
            } else {
                rvx = get_arg_val<uint32_t>(RT_PEERS + 2 * root_row + 0);
                rvy = get_arg_val<uint32_t>(RT_PEERS + 2 * root_row + 1);
                dst = get_write_ptr(cb_h_local) + my_row * slice_bytes;
                bytes = slice_bytes;
            }
            noc_async_write(get_read_ptr(cb_h_slice), get_noc_addr(rvx, rvy, dst), bytes);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(rvx, rvy, static_cast<uint32_t>(get_semaphore(SEM_HSLICE))), 1);
            noc_async_atomic_barrier();
            cb_pop_front(cb_h_slice, sl_a);
        }

        // Whole-round NoC1 ownership.  The receiver readers reserve/ack HACK_AHEAD slots, so this
        // diagonal writer can launch its round as soon as every destination has acknowledged it;
        // it need not wait for the readers to reach r in program order.  That is what overlaps
        // round 1/4/7 with the intervening NoC0 rounds instead of merely moving serial traffic.
        if (wd_mrow && !wd_mgroup && is_row_agg && h_round_on_writer(my_row)) {
            MaybeDeviceZoneScope("writer_hsend");
            moe_fused_swiglu::sem_wait_min(SEM_HSLICE, (b + 1) * HGROUPS);
            moe_fused_swiglu::sem_wait_min(SEM_H_FREE, (b + 1) * NUM_CORES);

            const uint32_t flag_slot = (b * KGROUPS + my_row) % DEPTH_H;
            // The ordinary mrow path publishes one payload-free alignment slot after its eight
            // rows, so every block's physical cb_h write pointer restarts at the base.  Round r's
            // physical row is therefore r % DEPTH_H even though its VALID flag rotates globally.
            const uint32_t hdst = get_write_ptr(cb_h) + (my_row % DEPTH_H) * HID_T * H_TILE;
            noc_async_read(get_noc_addr(get_write_ptr(cb_h_local)), hdst, HID_T * H_TILE);
            noc_async_read_barrier();
            h_slot_send_posted_noc1(flag_slot, hdst, HID_T * H_TILE);

            asm volatile("fence" ::: "memory");
            mailbox_words[moe_fused_swiglu::MBOX_HSEND_DONE] = b + 1;
        }

        // ---- output write-back, coalesced over the emb axis ----
        // EC_MAX is the L1 row stride of the block (uniform CB increment); `ec` is how many of
        // those columns this core actually owns.
        {
            MaybeDeviceZoneScope("writer_out_issue");
            {
                const uint32_t rp = get_read_ptr(cb_out_tiles);
                for (uint32_t t = 0; t < out_rows; ++t) {
                    // Full-M mrow compute publishes one row at a time; the ordinary/tail path
                    // publishes the whole block and therefore satisfies every cumulative wait at
                    // once. Do not pop here: issued DMAs still read this CB until the existing
                    // deferred barrier at the next block (or the epilogue) proves completion.
                    cb_wait_front(cb_out_tiles, (t + 1) * out_ec_max);
                    const uint32_t row = b * M_BLOCK + out_row_in_block + t;
                    if (row >= m_t) {
                        break;  // rows past ceil_tile(count) are never written
                    }
                    // Region-relative row -> destination row. The bound is what keeps a bad region
                    // base from scribbling over another expert's rows (or past the buffer): loud
                    // under --dev, silent-but-safe otherwise.
                    const uint32_t dst_row = out_row_base + row;
                    ASSERT(dst_row < OUT_M_T);
                    if (dst_row >= OUT_M_T) {
                        break;
                    }
                    BR::write(
                        out_acc,
                        dst_row * EMB_T,
                        out_jstart,
                        out_jstart + out_ec,
                        rp + t * out_ec_max * OUT_TILE,
                        OUT_TILE);
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
