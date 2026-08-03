// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// REALISTIC DM RECREATION — the op's data movement with BARRIERS WHERE THEY ARE ACTUALLY NEEDED.
//
// The floors measured by dl_stream / dl_collective put every transfer in flight and closed with ONE
// barrier, which is the most favourable schedule that exists. This kernel keeps the same traffic and
// the same per-core slices but restores the op's real DEPENDENCY ORDER, so the gap to those floors is
// the cost of the ordering itself. Still NO compute.
//
// Per M-block, in the op's own sequence (STAGE 1):
//
//   1. x stage      DRAM sub-page reads of my tile-row      -> BARRIER (must land before the mcast)
//   2. x mcast      row broadcast of the staged tile-row    -> BARRIER (would gate the matmul)
//   3. W chunks     GU_CHUNKS reads, ONE BARRIER PER CHUNK  (chunk c must land before matmul c —
//                   this is the barrier the one-barrier floor removes, and PERF 8 shows the schedule
//                   around it is worth 17 %)
//   4. reduce       column scatter -> BARRIER -> gather to root -> BARRIER
//   5. h rounds     HGROUPS rounds; the root's mcast closes with a barrier each round
//   6. W_down       one K-block prefetched per round (WD_AHEAD=1), barrier per round
//   7. out write    the block's output back to DRAM         -> BARRIER
//
// STAGE 0 collapses 3-7 into a single trailing barrier (the old floor) so the two can be A/B'd in one
// binary. What is still absent at STAGE 1 is the SEMAPHORE RENDEZVOUS that orders round r+1 behind
// round r across cores; adding it is stage 2 and is what should close the remaining gap to the op.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t STAGE = get_compile_time_arg_val(0);      // 0 = one trailing barrier, 1 = per-dependency
constexpr uint32_t IS_READER = get_compile_time_arg_val(1);  // 1 = NCRISC/NOC_0, 0 = BRISC/NOC_1
constexpr uint32_t HGROUPS = get_compile_time_arg_val(2);
constexpr uint32_t KGROUPS = get_compile_time_arg_val(3);
constexpr uint32_t GU_CHUNKS = get_compile_time_arg_val(4);
constexpr uint32_t HID_T = get_compile_time_arg_val(5);
constexpr uint32_t EMB_T = get_compile_time_arg_val(6);
constexpr uint32_t BFP4_TILE = get_compile_time_arg_val(7);
constexpr uint32_t BFP8_TILE = get_compile_time_arg_val(8);
constexpr uint32_t NUM_CORES = get_compile_time_arg_val(9);
constexpr uint32_t X_PAGE = get_compile_time_arg_val(10);
constexpr uint32_t CB_W = get_compile_time_arg_val(11);     // weight landing
constexpr uint32_t CB_SRC = get_compile_time_arg_val(12);   // collective payload source
constexpr uint32_t CB_LAND = get_compile_time_arg_val(13);  // collective landing
constexpr uint32_t CB_OUT = get_compile_time_arg_val(14);   // output staging
constexpr uint32_t TILE_H = 32;

constexpr uint32_t TA_BASE = 15;
constexpr auto w_args = TensorAccessorArgs<TA_BASE>();
constexpr auto wd_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr auto x_args = TensorAccessorArgs<wd_args.next_compile_time_args_offset()>();
constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

// A barrier that only exists at STAGE 1 — at STAGE 0 the work coalesces into one trailing barrier.
FORCE_INLINE void dep_read_barrier() {
    if constexpr (STAGE >= 1) {
        noc_async_read_barrier();
    }
}
FORCE_INLINE void dep_write_barrier() {
    if constexpr (STAGE >= 1) {
        noc_async_write_barrier();
    }
}

void kernel_main() {
    uint32_t i = 0;
    const uint32_t w_addr = get_arg_val<uint32_t>(i++);   // W_gate (reader) or W_up (writer)
    const uint32_t wd_addr = get_arg_val<uint32_t>(i++);  // W_down
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t o_addr = get_arg_val<uint32_t>(i++);
    const uint32_t my_col = get_arg_val<uint32_t>(i++);
    const uint32_t my_row = get_arg_val<uint32_t>(i++);
    const uint32_t m_eff = get_arg_val<uint32_t>(i++);
    const uint32_t kstart = get_arg_val<uint32_t>(i++);
    const uint32_t kr = get_arg_val<uint32_t>(i++);
    const uint32_t hstart = get_arg_val<uint32_t>(i++);
    const uint32_t hn = get_arg_val<uint32_t>(i++);
    const uint32_t ecstart = get_arg_val<uint32_t>(i++);
    const uint32_t ec = get_arg_val<uint32_t>(i++);
    const uint32_t x_rows = get_arg_val<uint32_t>(i++);
    const uint32_t x_row0 = get_arg_val<uint32_t>(i++);
    const uint32_t is_root = get_arg_val<uint32_t>(i++);
    const uint32_t row_a0 = get_arg_val<uint32_t>(i++);  // row rect, already in this NoC's order
    const uint32_t row_b0 = get_arg_val<uint32_t>(i++);
    const uint32_t row_a1 = get_arg_val<uint32_t>(i++);
    const uint32_t row_b1 = get_arg_val<uint32_t>(i++);
    const uint32_t all_a0 = get_arg_val<uint32_t>(i++);  // whole-grid rect, this NoC's order
    const uint32_t all_b0 = get_arg_val<uint32_t>(i++);
    const uint32_t all_a1 = get_arg_val<uint32_t>(i++);
    const uint32_t all_b1 = get_arg_val<uint32_t>(i++);
    const uint32_t col_base = i;  // KGROUPS (x, y) pairs, root first

    const auto w_acc = TensorAccessor(w_args, w_addr, BFP4_TILE);
    const auto wd_acc = TensorAccessor(wd_args, wd_addr, BFP4_TILE);
    const uint32_t src = get_write_ptr(CB_SRC);
    const uint32_t land = get_write_ptr(CB_LAND);
    const uint32_t wp = get_write_ptr(CB_W);

    // ---- 1/2. x staging then its row multicast (reader only, injector cores only) ----
    if constexpr (IS_READER) {
        if (x_rows) {
            const auto x_acc = TensorAccessor(x_args, x_addr, X_PAGE);
            const uint32_t slice = kr * TILE_H * 2;
            for (uint32_t r = 0; r < x_rows; ++r) {
                for (uint32_t s = 0; s < TILE_H; ++s) {
                    noc_async_read(
                        x_acc.get_noc_addr((x_row0 + r) * TILE_H + s, kstart * TILE_H * 2),
                        land + (r * TILE_H + s) * slice,
                        slice);
                }
            }
            // NEEDED: the sticks must be in L1 before they can be broadcast.
            noc_async_read_barrier();
            const uint32_t xbytes = kr * BFP8_TILE;
            for (uint32_t r = 0; r < x_rows; ++r) {
                const uint64_t dst = get_noc_multicast_addr(row_a0, row_b0, row_a1, row_b1, land + r * xbytes);
                noc_async_write_multicast(land + r * xbytes, dst, xbytes, HGROUPS - 1, /*linked=*/false);
            }
            dep_write_barrier();  // NEEDED at stage 1: would gate the matmul on the receivers
        }
    }

    // ---- 3. the weight stream, ONE BARRIER PER GU CHUNK ----
    {
        const uint32_t chunk_w = hn / GU_CHUNKS ? hn / GU_CHUNKS : 1;
        for (uint32_t c = 0; c < GU_CHUNKS; ++c) {
            const uint32_t h0 = c * chunk_w;
            if (h0 >= hn) {
                break;
            }
            uint32_t w = hn - h0;
            if (w > chunk_w) {
                w = chunk_w;
            }
            for (uint32_t k = 0; k < kr; ++k) {
                noc_async_read(
                    w_acc.get_noc_addr((kstart + k) * HID_T + hstart + h0),
                    wp + (c * kr + k) * chunk_w * BFP4_TILE,
                    w * BFP4_TILE);
            }
            dep_read_barrier();  // NEEDED at stage 1: chunk c must land before matmul c
        }
    }

    // ---- 4. column reduce-scatter: scatter, barrier, gather to root, barrier ----
    {
        const uint32_t block = m_eff * hn * BFP8_TILE;
        const uint32_t slice = block / KGROUPS;
        for (uint32_t o = 0; o < KGROUPS; ++o) {
            const uint32_t ox = get_arg_val<uint32_t>(col_base + 2 * o);
            const uint32_t oy = get_arg_val<uint32_t>(col_base + 2 * o + 1);
            noc_async_write(src + o * slice, get_noc_addr(ox, oy, land + my_row * slice), slice);
        }
        dep_write_barrier();  // NEEDED: the owner cannot reduce a slice that has not arrived
        const uint32_t rx = get_arg_val<uint32_t>(col_base + 0);
        const uint32_t ry = get_arg_val<uint32_t>(col_base + 1);
        noc_async_write(src, get_noc_addr(rx, ry, land + my_row * slice), slice);
        dep_write_barrier();  // NEEDED: the root's h is not complete until every owner has shipped
    }

    // ---- 5/6. h rounds, and W_down prefetched one K-block per round (WD_AHEAD=1) ----
    {
        const uint32_t hbytes = m_eff * hn * BFP8_TILE;
        const uint32_t kb = HID_T / HGROUPS ? HID_T / HGROUPS : 1;  // W_down K-block per round
        for (uint32_t r = 0; r < HGROUPS; ++r) {
            if (is_root && r == my_col) {
                const uint64_t dst = get_noc_multicast_addr(all_a0, all_b0, all_a1, all_b1, land);
                noc_async_write_multicast(land, dst, hbytes, NUM_CORES - 1, /*linked=*/false);
                dep_write_barrier();  // NEEDED: receivers key on this round's arrival
            }
            if constexpr (!IS_READER) {  // W_down rides the writer, like the op
                const uint32_t j0 = r * kb;
                for (uint32_t j = 0; j < kb && (j0 + j) < HID_T; ++j) {
                    noc_async_read(
                        wd_acc.get_noc_addr((j0 + j) * EMB_T + ecstart),
                        wp + ((j0 + j) % kb) * ec * BFP4_TILE,
                        ec * BFP4_TILE);
                }
                dep_read_barrier();  // NEEDED: this round's down block must land before it is consumed
            }
        }
    }

    // ---- 7. the block's output back to DRAM (writer only, like the op) ----
    if constexpr (!IS_READER) {
        const auto o_acc = TensorAccessor(o_args, o_addr, BFP8_TILE);
        const uint32_t obase = get_write_ptr(CB_OUT);
        for (uint32_t t = 0; t < m_eff * ec; ++t) {
            noc_async_write(obase + t * BFP8_TILE, o_acc.get_noc_addr(t + m_eff * ecstart), BFP8_TILE);
        }
    }

    // Whatever the stage, everything must be closed before the kernel exits.
    noc_async_read_barrier();
    noc_async_write_barrier();
}
