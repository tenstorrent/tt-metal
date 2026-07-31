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

constexpr uint32_t cb_x_in = get_compile_time_arg_val(24);
constexpr uint32_t cb_x_tiles = get_compile_time_arg_val(25);
constexpr uint32_t cb_x_stage = get_compile_time_arg_val(26);
constexpr uint32_t cb_w_gate = get_compile_time_arg_val(27);
constexpr uint32_t cb_w_down = get_compile_time_arg_val(28);
constexpr uint32_t cb_reduce_gate_in = get_compile_time_arg_val(29);
constexpr uint32_t cb_reduce_up_in = get_compile_time_arg_val(30);
constexpr uint32_t cb_h = get_compile_time_arg_val(31);
constexpr uint32_t cb_h_local = get_compile_time_arg_val(32);
constexpr uint32_t cb_idx_scratch = get_compile_time_arg_val(33);
constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(34);

constexpr uint32_t CT_XMCAST = 35;
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

// ---------------------------------------------------------------------------
// Bank-contiguous run enumeration over an N-axis linear range.
//
// Interleaved page -> bank is `page_id % NUM_BANKS`, with in-bank slot `page_id / NUM_BANKS` at
// stride aligned_page_size. For every tensor here the row stride (HID_T / EMB_T) is a multiple of
// NUM_BANKS, so bank(row*stride + n) == n % NUM_BANKS: a stride-NUM_BANKS run of columns at a
// fixed row is physically contiguous inside ONE bank and reads as ONE transaction. `remap_n`
// re-indexes the logical N axis so that CONSECUTIVE linear indices walk one bank's slots.
// ---------------------------------------------------------------------------
FORCE_INLINE uint32_t remap_n(uint32_t j, uint32_t slots) {
    if constexpr (REMAP) {
        return (j / slots) + NUM_BANKS * (j % slots);
    } else {
        return j;
    }
}

// Length of the maximal bank-contiguous run starting at linear index j inside [j, end).
FORCE_INLINE uint32_t run_len(uint32_t j, uint32_t end, uint32_t slots) {
    if constexpr (REMAP) {
        uint32_t r = end - j;
        const uint32_t to_bank_edge = slots - (j % slots);
        if (to_bank_edge < r) {
            r = to_bank_edge;
        }
        if (WRUN < r) {
            r = WRUN;
        }
        return r;
    } else {
        return 1;
    }
}

constexpr uint32_t N_STRIDE = REMAP ? NUM_BANKS : 1;

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
    volatile tt_l1_ptr uint32_t* mbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_addr);
    mbox[0] = count;
    mbox[1] = m_t;
    mbox[2] = m_blocks;
    invalidate_l1_cache();
    mbox[3] = MAILBOX_MAGIC;

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

    constexpr uint32_t X_SLOT_TILES = M_BLOCK * KR_PAD;   // resident in0 block, one slot
    constexpr uint32_t WG_BLOCK_TILES = KR_PAD * HN_PAD;  // one gate weight K-block (num_k_blocks == 1)
    constexpr uint32_t H_BLOCK_TILES = M_BLOCK * HN_PAD;  // one phase-2 K-block of h
    constexpr uint32_t X_ROW_BYTES = KR_PAD * BFP8_TILE;

    // `count == 0` -> m_blocks == 0 on every core: no CB traffic, no collective round, no
    // semaphore. Uniform across the grid, so it cannot hang.
    for (uint32_t b = 0; b < m_blocks; ++b) {
        // -------------------------------------------------------------------
        // Phase 1a — stage x and multicast it along the grid row.
        //
        // cb_x_tiles is ONE slot of M_BLOCK*KR_PAD tiles, so its write pointer is the same L1
        // address on every core in the row (mcast_pipe requires an identical landing address).
        // -------------------------------------------------------------------
        cb_reserve_back(cb_x_tiles, X_SLOT_TILES);
        const uint32_t x_base = get_write_ptr(cb_x_tiles);

        for (uint32_t t = 0; t < M_BLOCK; ++t) {
            const uint32_t round = t % HGROUPS;
            const uint32_t dst = x_base + t * X_ROW_BYTES;
            if (round == my_col) {
                uint32_t row = b * M_BLOCK + t;
                if (row >= M_T_MAX) {
                    row = M_T_MAX - 1;  // rows past the sized region are UNDEFINED; stay in bounds
                }
                if constexpr (INPUT_FORMAT == 0) {
                    // bf16 ROW_MAJOR: read this row-group's emb slice of 32 sticks, compute
                    // tilizes it to bfp8 in cb_x_stage, then multicast (loopback into cb_x_tiles).
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
                    if constexpr (xmc.active) {
                        x_send.send(get_read_ptr(cb_x_stage), dst, X_ROW_BYTES);
                    } else {
                        noc_async_read(get_noc_addr(get_read_ptr(cb_x_stage)), dst, X_ROW_BYTES);
                        noc_async_read_barrier();
                    }
                    cb_pop_front(cb_x_stage, KR_PAD);
                } else {
                    // bfp8_b TILE: land the tiles straight in the resident slot, then broadcast
                    // in place (src == dst, so the pipe multicasts EXCLUDE-source).
                    for (uint32_t i = 0; i < kr; ++i) {
                        noc_async_read(x_acc.get_noc_addr(row * EMB_T + kstart + i), dst + i * BFP8_TILE, BFP8_TILE);
                    }
                    noc_async_read_barrier();
                    if constexpr (xmc.active) {
                        x_send.send(dst, dst, X_ROW_BYTES);
                    }
                }
            } else {
                if constexpr (xmc.active) {
                    x_recv.receive(round);
                }
            }
        }
        cb_push_back(cb_x_tiles, X_SLOT_TILES);

        // -------------------------------------------------------------------
        // Phase 1b — W_gate: one K-block == the whole per-row K extent, coalesced bank runs.
        // (W_up is the writer's twin on NoC1 — the dual-NoC split of op_design.md §1.5.)
        // -------------------------------------------------------------------
        cb_reserve_back(cb_w_gate, WG_BLOCK_TILES);
        {
            const uint32_t wp = get_write_ptr(cb_w_gate);
            for (uint32_t k = 0; k < kr; ++k) {
                const uint32_t kt = kstart + k;
                uint32_t j = hstart;
                uint32_t noff = 0;
                while (j < hstart + hn) {
                    const uint32_t len = run_len(j, hstart + hn, SLOTS_H);
                    const uint32_t first = remap_n(j, SLOTS_H);
                    noc_async_read(
                        wg_acc.get_noc_addr(kt * HID_T + first), wp + (k * HN_PAD + noff) * BFP4_TILE, len * BFP4_TILE);
                    j += len;
                    noff += len;
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_w_gate, WG_BLOCK_TILES);

        // -------------------------------------------------------------------
        // Phase 1c — reduce tree, PARENT side. One slot per incoming partial, so the child's
        // landing address is the CB base on every core; the invite (SEM_GO) is the flow control
        // that keeps a child from overwriting a slot compute has not consumed yet.
        // -------------------------------------------------------------------
        for (uint32_t c = 0; c < num_children; ++c) {
            const uint32_t cx = get_arg_val<uint32_t>(RT_CHILDREN + 2 * c + 0);
            const uint32_t cy = get_arg_val<uint32_t>(RT_CHILDREN + 2 * c + 1);
            cb_reserve_back(cb_reduce_gate_in, H_BLOCK_TILES);
            cb_reserve_back(cb_reduce_up_in, H_BLOCK_TILES);
            noc_semaphore_inc(get_noc_addr(cx, cy, static_cast<uint32_t>(get_semaphore(SEM_GO))), 1);
            noc_semaphore_wait_min(sem_data_ptr, ++data_arrivals);
            cb_push_back(cb_reduce_gate_in, H_BLOCK_TILES);
            cb_push_back(cb_reduce_up_in, H_BLOCK_TILES);
        }

        // -------------------------------------------------------------------
        // Phase 2 — W_down K-block r fused with round r of the h all-gather. The gather rides
        // the phase-2 K stream, so it overlaps `down` compute and flow-controls itself on cb_h.
        // -------------------------------------------------------------------
        // EC_MAX-wide K-block so the CB increment is uniform across cores; a core with
        // ec < EC_MAX leaves the tail columns unwritten and the matmul never reads them.
        constexpr uint32_t wd_block_tiles = HN_PAD * EC_MAX;
        for (uint32_t r = 0; r < HGROUPS; ++r) {
            const uint32_t hbase = r * HN_PAD;
            uint32_t hn_r = HN_PAD;
            if (hbase + hn_r > HID_T) {
                hn_r = HID_T - hbase;
            }

            cb_reserve_back(cb_w_down, wd_block_tiles);
            {
                const uint32_t wp = get_write_ptr(cb_w_down);
                for (uint32_t k = 0; k < hn_r; ++k) {
                    const uint32_t ht = remap_n(hbase + k, SLOTS_H);
                    uint32_t j = jstart;
                    uint32_t eoff = 0;
                    while (j < jstart + ec) {
                        const uint32_t len = run_len(j, jstart + ec, SLOTS_E);
                        const uint32_t first = remap_n(j, SLOTS_E);
                        noc_async_read(
                            wd_acc.get_noc_addr(ht * EMB_T + first),
                            wp + (k * EC_MAX + eoff) * BFP4_TILE,
                            len * BFP4_TILE);
                        j += len;
                        eoff += len;
                    }
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb_w_down, wd_block_tiles);

            cb_reserve_back(cb_h, H_BLOCK_TILES);
            const uint32_t hdst = get_write_ptr(cb_h);
            if (is_root && r == my_col) {
                cb_wait_front(cb_h_local, H_BLOCK_TILES);
                if constexpr (hmc.active) {
                    h_send.send(get_read_ptr(cb_h_local), hdst, H_BLOCK_TILES * BFP8_TILE);
                } else {
                    noc_async_read(get_noc_addr(get_read_ptr(cb_h_local)), hdst, H_BLOCK_TILES * BFP8_TILE);
                    noc_async_read_barrier();
                }
                cb_pop_front(cb_h_local, H_BLOCK_TILES);
            } else {
                if constexpr (hmc.active) {
                    // Round r's sender is column r's root, core (r, r % KGROUPS); the rotating
                    // sender list is row-major over the rect.
                    h_recv.receive((r % KGROUPS) * HGROUPS + r);
                }
            }
            cb_push_back(cb_h, H_BLOCK_TILES);
        }
    }
}
