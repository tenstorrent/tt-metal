// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// gateup_reduce_overlap — READER (NoC0): x + W_gate loads, and the PARENT side of the reduce tree.
//
// Loading is a "prep" phase, zoned separately and IDENTICAL across every variant (baseline /
// split-serial / split-pipelined / HN-axis / M-axis) — it is not what this bench measures, so it is
// held deliberately simple (plain per-tile noc_async_read, no bank-run coalescing; that is a
// different, already-characterised part of the op, see op_design.md §1.5). All of a stage's data is
// staged BEFORE the reduce loop starts so the matmul+reduce schedule (the actual subject) is not
// entangled with DRAM latency.
//
// The reduce transport reproduces the real op's mechanism exactly (moe_fused_swiglu_reader.cpp
// phase 1c): unicast + a monotone counting semaphore pair (SEM_GO invite, SEM_DATA arrival),
// REDUCE_SLOTS==1 (one child in flight at a time — Refinement 2 lever 1 measured >1 as a
// regression), generalised over S stages with one cumulative `data_arrivals` counter spanning the
// whole per-core kernel (same "monotone, no reset" invariant as the real op).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

constexpr uint32_t KR_PAD = get_compile_time_arg_val(0);
constexpr uint32_t HN_PAD = get_compile_time_arg_val(1);
constexpr uint32_t M_EFF = get_compile_time_arg_val(2);
constexpr uint32_t S = get_compile_time_arg_val(3);
constexpr uint32_t SPLIT_AXIS = get_compile_time_arg_val(4);
constexpr uint32_t HN_BLOCK = get_compile_time_arg_val(5);
constexpr uint32_t M_GROUP = get_compile_time_arg_val(6);
constexpr uint32_t EMB_T = get_compile_time_arg_val(7);  // total K width of x/w_gate/w_up, in tiles
constexpr uint32_t MAX_CHILDREN = get_compile_time_arg_val(8);
constexpr uint32_t SEM_GO = get_compile_time_arg_val(9);
constexpr uint32_t SEM_DATA = get_compile_time_arg_val(10);
constexpr uint32_t BFP8_TILE = get_compile_time_arg_val(11);
constexpr uint32_t BFP4_TILE = get_compile_time_arg_val(12);
constexpr uint32_t CB_X_BASE = get_compile_time_arg_val(13);
constexpr uint32_t CB_WG_BASE = get_compile_time_arg_val(14);
constexpr uint32_t CB_REDUCE_GATE_BASE = get_compile_time_arg_val(15);
constexpr uint32_t CB_REDUCE_UP_BASE = get_compile_time_arg_val(16);

constexpr uint32_t TA_BASE = 17;
constexpr auto x_args = TensorAccessorArgs<TA_BASE>();
constexpr auto wg_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

constexpr uint32_t STAGE_ROWS = (SPLIT_AXIS == 1) ? M_GROUP : M_EFF;
constexpr uint32_t STAGE_COLS = (SPLIT_AXIS == 0) ? HN_BLOCK : HN_PAD;
constexpr uint32_t STAGE_TILES = STAGE_ROWS * STAGE_COLS;

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t wg_addr = get_arg_val<uint32_t>(1);
    const uint32_t kr = get_arg_val<uint32_t>(2);
    const uint32_t kstart = get_arg_val<uint32_t>(3);
    const uint32_t num_children = get_arg_val<uint32_t>(4);

    const auto x_acc = TensorAccessor(x_args, x_addr, BFP8_TILE);
    const auto wg_acc = TensorAccessor(wg_args, wg_addr, BFP4_TILE);

    {
        MaybeDeviceZoneScope("reader_prep");
        // ---- x: HN-split shares ALL M_EFF rows across every stage (read once); M-split gives
        // each stage its OWN M_GROUP-row slice (S dedicated reads). Row-major inside the slot,
        // KR_PAD-strided per row (only `kr` of them real — matmul's KrSteps bounds the FMA loop). ----
        if constexpr (SPLIT_AXIS == 0) {
            cb_reserve_back(CB_X_BASE, M_EFF * KR_PAD);
            const uint32_t wp = get_write_ptr(CB_X_BASE);
            for (uint32_t m = 0; m < M_EFF; ++m) {
                for (uint32_t k = 0; k < kr; ++k) {
                    noc_async_read(
                        x_acc.get_noc_addr(m * EMB_T + kstart + k), wp + (m * KR_PAD + k) * BFP8_TILE, BFP8_TILE);
                }
            }
            noc_async_read_barrier();
            cb_push_back(CB_X_BASE, M_EFF * KR_PAD);
        } else {
            for (uint32_t s = 0; s < S; ++s) {
                const uint32_t xcb = CB_X_BASE + s;
                cb_reserve_back(xcb, STAGE_ROWS * KR_PAD);
                const uint32_t wp = get_write_ptr(xcb);
                for (uint32_t m = 0; m < STAGE_ROWS; ++m) {
                    const uint32_t row = s * STAGE_ROWS + m;
                    for (uint32_t k = 0; k < kr; ++k) {
                        noc_async_read(
                            x_acc.get_noc_addr(row * EMB_T + kstart + k), wp + (m * KR_PAD + k) * BFP8_TILE, BFP8_TILE);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(xcb, STAGE_ROWS * KR_PAD);
            }
        }

        // ---- W_gate: HN-split gives each stage its OWN HN_BLOCK-wide column slice (S dedicated
        // reads); M-split shares the full HN_PAD-wide block across every stage (read once). ----
        if constexpr (SPLIT_AXIS == 0) {
            for (uint32_t s = 0; s < S; ++s) {
                const uint32_t wgcb = CB_WG_BASE + s;
                cb_reserve_back(wgcb, KR_PAD * STAGE_COLS);
                const uint32_t wp = get_write_ptr(wgcb);
                for (uint32_t k = 0; k < kr; ++k) {
                    for (uint32_t n = 0; n < STAGE_COLS; ++n) {
                        noc_async_read(
                            wg_acc.get_noc_addr((kstart + k) * HN_PAD + s * STAGE_COLS + n),
                            wp + (k * STAGE_COLS + n) * BFP4_TILE,
                            BFP4_TILE);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(wgcb, KR_PAD * STAGE_COLS);
            }
        } else {
            cb_reserve_back(CB_WG_BASE, KR_PAD * HN_PAD);
            const uint32_t wp = get_write_ptr(CB_WG_BASE);
            for (uint32_t k = 0; k < kr; ++k) {
                for (uint32_t n = 0; n < HN_PAD; ++n) {
                    noc_async_read(
                        wg_acc.get_noc_addr((kstart + k) * HN_PAD + n), wp + (k * HN_PAD + n) * BFP4_TILE, BFP4_TILE);
                }
            }
            noc_async_read_barrier();
            cb_push_back(CB_WG_BASE, KR_PAD * HN_PAD);
        }
    }

    // ---- reduce tree, PARENT side, per stage. Skipped entirely (num_children == 0) on a leaf. ----
    if (num_children > 0) {
        uint32_t children_x[MAX_CHILDREN];
        uint32_t children_y[MAX_CHILDREN];
        for (uint32_t c = 0; c < num_children; ++c) {
            children_x[c] = get_arg_val<uint32_t>(5 + 2 * c + 0);
            children_y[c] = get_arg_val<uint32_t>(5 + 2 * c + 1);
        }
        const uint32_t sem_go_addr = static_cast<uint32_t>(get_semaphore(SEM_GO));
        volatile tt_l1_ptr uint32_t* sem_data_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_DATA)));
        uint32_t data_arrivals = 0;

        // REDUCE_SLOTS == 1 (the real op's Phase-0 / shipped protocol — lever 1 in changelog.md
        // Refinement 2 measured >1 concurrent slot as a regression). The CB holds exactly ONE
        // child's slot (STAGE_TILES, no extra depth), so reserve/invite/wait/push must happen ONCE
        // PER CHILD, not once for the whole stage: `cb_reserve_back` for child c+1 blocks until
        // compute has popped child c's data, which is the flow control the whole transport rests
        // on. Collapsing all children into one reserve/push (as an earlier draft of this file did)
        // lets child c+1 overwrite child c's slot before compute ever consumes it, and leaves
        // compute's later per-child waits with no further push coming — a real hang, not a stall.
        MaybeDeviceZoneScope("reader_reduce");
        for (uint32_t s = 0; s < S; ++s) {
            const uint32_t rg = CB_REDUCE_GATE_BASE + s;
            const uint32_t ru = CB_REDUCE_UP_BASE + s;
            for (uint32_t c = 0; c < num_children; ++c) {
                cb_reserve_back(rg, STAGE_TILES);
                cb_reserve_back(ru, STAGE_TILES);
                noc_semaphore_inc(get_noc_addr(children_x[c], children_y[c], sem_go_addr), 1);
                data_arrivals += 1;
                noc_semaphore_wait_min(sem_data_ptr, data_arrivals);
                cb_push_back(rg, STAGE_TILES);
                cb_push_back(ru, STAGE_TILES);
            }
        }
    }
}
