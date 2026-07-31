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
//   4. the PARENT side of the gate/up cross-column reduce tree (invite child -> land its
//      partials in cb_reduce_*_in -> publish to compute);
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

#include "moe_fused_swiglu_bank_runs.hpp"  // the ONE definition of the bank-run coalescing
#include "moe_fused_swiglu_common.hpp"     // the ONE definition of the mailbox word layout

using namespace dataflow_kernel_lib;

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

constexpr uint32_t cb_x_in = get_compile_time_arg_val(26);
constexpr uint32_t cb_x_tiles = get_compile_time_arg_val(27);
constexpr uint32_t cb_x_stage = get_compile_time_arg_val(28);
constexpr uint32_t cb_w_gate = get_compile_time_arg_val(29);
constexpr uint32_t cb_w_down = get_compile_time_arg_val(30);
constexpr uint32_t cb_reduce_gate_in = get_compile_time_arg_val(31);
constexpr uint32_t cb_reduce_up_in = get_compile_time_arg_val(32);
constexpr uint32_t cb_h = get_compile_time_arg_val(33);
constexpr uint32_t cb_h_local = get_compile_time_arg_val(34);
constexpr uint32_t cb_idx_scratch = get_compile_time_arg_val(35);
constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(36);

constexpr uint32_t CT_XMCAST = 37;
constexpr uint32_t CT_HMCAST = CT_XMCAST + 5;

constexpr uint32_t TILE_H = 32;
constexpr uint32_t BF16_TILE_ROW_BYTES = TILE_H * 2;  // one 32-element tile slice of a bf16 stick

// runtime-arg layout
constexpr uint32_t RT_CHILDREN = 15;
constexpr uint32_t RT_XMCAST = RT_CHILDREN + 2 * MAX_CHILDREN;
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

    constexpr uint32_t SLOTS_H = REMAP ? (HID_T / NUM_BANKS) : HID_T;
    constexpr uint32_t SLOTS_E = REMAP ? (EMB_T / NUM_BANKS) : EMB_T;

    constexpr uint32_t WG_BLOCK_TILES = KR_PAD * HN_PAD;      // one gate weight K-block (num_k_blocks == 1)
    constexpr uint32_t REDUCE_SLOT_TILES = M_BLOCK * HN_PAD;  // ONE whole slot — see the note in phase 1c
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

        // -------------------------------------------------------------------
        // Phase 1a — stage x and multicast it along the grid row.
        //
        // cb_x_tiles is ONE slot of M_BLOCK*KR_PAD tiles, so its write pointer is the same L1
        // address on every core in the row (mcast_pipe requires an identical landing address).
        // -------------------------------------------------------------------
        // W_gate is ISSUED HERE, ahead of the x rounds, and only BARRIERED/pushed after them:
        // the x multicast is a chain of M_BLOCK semaphore handshakes with almost no NoC payload,
        // so the whole gate weight block's DRAM latency hides underneath it. With
        // num_k_blocks == 1 the block has to be fully fronted before the gate matmul starts, so
        // this is the only place its latency can be overlapped.
        cb_reserve_back(cb_w_gate, WG_BLOCK_TILES);
        const uint32_t wg_wp = get_write_ptr(cb_w_gate);
        for (uint32_t k = 0; k < kr; ++k) {
            BR::read(
                wg_acc, (kstart + k) * HID_T, hstart, hstart + hn, SLOTS_H, wg_wp + k * HN_PAD * BFP4_TILE, BFP4_TILE);
        }

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
                for (uint32_t s = 0; s < TILE_H; ++s) {
                    noc_async_read(
                        x_acc.get_noc_addr(row * TILE_H + s, kstart * BF16_TILE_ROW_BYTES),
                        wp + s * X_SLICE,
                        kr * BF16_TILE_ROW_BYTES);
                }
                noc_async_read_barrier();
                cb_push_back(cb_x_in, TILE_H);

                cb_wait_front(cb_x_stage, KR_PAD);
                noc_async_read(get_noc_addr(get_read_ptr(cb_x_stage)), dst, X_ROW_BYTES);
                noc_async_read_barrier();
                cb_pop_front(cb_x_stage, KR_PAD);
            } else {
                // bfp8_b TILE: the tiles land straight in the resident slot, no tilize.
                for (uint32_t i = 0; i < kr; ++i) {
                    noc_async_read(x_acc.get_noc_addr(row * EMB_T + kstart + i), dst + i * BFP8_TILE, BFP8_TILE);
                }
                noc_async_read_barrier();
            }
        }

        // ---- x multicast chain ----
        // m_eff rounds, not M_BLOCK: at count 128 (M_t 4) this is HALF the handshake chain and half
        // the staged bytes, and at count 32 an eighth. m_eff divides M_BLOCK, so cb_x_tiles' write
        // pointer stays block-aligned and identical on every core in the row (which mcast_pipe
        // requires of the landing address).
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
        cb_push_back(cb_x_tiles, x_slot_tiles);

        // -------------------------------------------------------------------
        // Phase 1b — W_gate landed under the x rounds; publish it.
        // (W_up is the writer's twin on NoC1 — the dual-NoC split of op_design.md §1.5.)
        // -------------------------------------------------------------------
        noc_async_read_barrier();
        cb_push_back(cb_w_gate, WG_BLOCK_TILES);

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
            const uint32_t wp = get_write_ptr(cb_w_down);
            for (uint32_t r = 0; r < WD_AHEAD; ++r) {
                const uint32_t hbase = r * HN_PAD;
                uint32_t hn_r = HN_PAD;
                if (hbase + hn_r > HID_T) {
                    hn_r = HID_T - hbase;
                }
                for (uint32_t k = 0; k < hn_r; ++k) {
                    // W_down's K axis is `h`'s hidden axis, which the Hn split remapped too, so the
                    // row index goes through the same remap as the N axis.
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
        }

        // -------------------------------------------------------------------
        // Phase 1c — reduce tree, PARENT side. One slot per incoming partial, so the child's
        // landing address is the CB base on every core; the invite (SEM_GO) is the flow control
        // that keeps a child from overwriting a slot compute has not consumed yet.
        //
        // These two CBs are the ONE place the m_eff shrink does NOT move the CB accounting: the
        // child unicasts to its OWN `get_write_ptr(cb_reduce_*_in)` on the assumption that the
        // parent's pointer is the same, and that only holds while every push is a WHOLE slot (a
        // whole-slot push always wraps back to the CB base, on every core, whatever its child
        // count). So the slot stays full here and the child writes only the m_eff*HN_PAD tiles that
        // carry live tokens; compute adds those and drops the rest (they are undefined rows).
        // The transport — the expensive part, up to 4 serial ~102 KB round trips — halves anyway.
        // -------------------------------------------------------------------
        for (uint32_t c = 0; c < num_children; ++c) {
            const uint32_t cx = get_arg_val<uint32_t>(RT_CHILDREN + 2 * c + 0);
            const uint32_t cy = get_arg_val<uint32_t>(RT_CHILDREN + 2 * c + 1);
            cb_reserve_back(cb_reduce_gate_in, REDUCE_SLOT_TILES);
            cb_reserve_back(cb_reduce_up_in, REDUCE_SLOT_TILES);
            noc_semaphore_inc(get_noc_addr(cx, cy, static_cast<uint32_t>(get_semaphore(SEM_GO))), 1);
            noc_semaphore_wait_min(sem_data_ptr, ++data_arrivals);
            cb_push_back(cb_reduce_gate_in, REDUCE_SLOT_TILES);
            cb_push_back(cb_reduce_up_in, REDUCE_SLOT_TILES);
        }

        // -------------------------------------------------------------------
        // Phase 2 — round r of the h all-gather, with W_down already in flight. The gather rides
        // the phase-2 K stream, so it overlaps `down` compute and flow-controls itself on cb_h.
        // The batch issued above landed under the reduce handshakes; publish it, then stream the
        // remaining K-blocks one round ahead of the round that consumes them.
        // -------------------------------------------------------------------
        noc_async_read_barrier();
        cb_push_back(cb_w_down, WD_AHEAD * WD_BLOCK_TILES);
        for (uint32_t r = 0; r < HGROUPS; ++r) {
            // This round's cb_h slot is reserved BEFORE the W_down prefetch so that the round's
            // sender can ISSUE its self-copy first and ONE barrier below covers both: the copy then
            // rides the W_down DRAM read's latency instead of adding its own to the round chain.
            // Reordering the two reserves cannot deadlock — cb_w_down always runs WD_AHEAD blocks
            // ahead of cb_h, so a compute thread waiting on K-block k already has its in1.
            cb_reserve_back(cb_h, h_block_tiles);
            const uint32_t hdst = get_write_ptr(cb_h);
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
                cb_wait_front(cb_h_local, h_block_tiles);
                noc_async_read(get_noc_addr(get_read_ptr(cb_h_local)), hdst, h_block_tiles * BFP8_TILE);
            }

            // Stay WD_AHEAD K-blocks ahead of the round being consumed.
            const uint32_t pre = r + WD_AHEAD;
            const bool prefetch = pre < HGROUPS;
            if (prefetch) {
                const uint32_t hbase = pre * HN_PAD;
                uint32_t hn_r = HN_PAD;
                if (hbase + hn_r > HID_T) {
                    hn_r = HID_T - hbase;
                }
                cb_reserve_back(cb_w_down, WD_BLOCK_TILES);
                const uint32_t wp = get_write_ptr(cb_w_down);
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
            noc_async_read_barrier();  // ONE barrier: the sender's h self-copy AND the W_down block
            if (prefetch) {
                cb_push_back(cb_w_down, WD_BLOCK_TILES);
            }

            if (i_send) {
                cb_pop_front(cb_h_local, h_block_tiles);
                if constexpr (hmc.active) {
                    h_send.send(hdst, hdst, h_block_tiles * BFP8_TILE);
                }
            } else {
                if constexpr (hmc.active) {
                    // Round r's sender is column r's root, core (r, r % KGROUPS); the rotating
                    // sender list is row-major over the rect.
                    h_recv.receive((r % KGROUPS) * HGROUPS + r);
                }
            }
            cb_push_back(cb_h, h_block_tiles);
        }
    }
}
