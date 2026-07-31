// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// gateup_reduce_overlap — WRITER (NoC1): W_up load, the CHILD side of the reduce tree (non-root),
// and the final output write-back (root only).
//
// Mirrors the real op's writer (moe_fused_swiglu_writer.cpp reduce_child): the child's own
// write-pointer for its per-stage reduce-landing CB IS the parent's landing address (every core has
// the identical CB layout, and the reduce CBs are always pushed whole), so there is no address
// negotiation — see moe_fused_swiglu_writer.cpp's comment on this same trick.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

constexpr uint32_t KR_PAD = get_compile_time_arg_val(0);
constexpr uint32_t HN_PAD = get_compile_time_arg_val(1);
constexpr uint32_t M_EFF = get_compile_time_arg_val(2);
constexpr uint32_t S = get_compile_time_arg_val(3);
constexpr uint32_t SPLIT_AXIS = get_compile_time_arg_val(4);
constexpr uint32_t HN_BLOCK = get_compile_time_arg_val(5);
constexpr uint32_t M_GROUP = get_compile_time_arg_val(6);
constexpr uint32_t EMB_T = get_compile_time_arg_val(7);
constexpr uint32_t SEM_GO = get_compile_time_arg_val(8);
constexpr uint32_t SEM_DATA = get_compile_time_arg_val(9);
constexpr uint32_t BFP8_TILE = get_compile_time_arg_val(10);
constexpr uint32_t BFP4_TILE = get_compile_time_arg_val(11);
constexpr uint32_t CB_WU_BASE = get_compile_time_arg_val(12);
constexpr uint32_t CB_A_BASE = get_compile_time_arg_val(13);
constexpr uint32_t CB_B_BASE = get_compile_time_arg_val(14);
constexpr uint32_t CB_REDUCE_GATE_BASE = get_compile_time_arg_val(15);
constexpr uint32_t CB_REDUCE_UP_BASE = get_compile_time_arg_val(16);

constexpr uint32_t TA_BASE = 17;
constexpr auto wu_args = TensorAccessorArgs<TA_BASE>();
constexpr auto out_args = TensorAccessorArgs<wu_args.next_compile_time_args_offset()>();

constexpr uint32_t STAGE_ROWS = (SPLIT_AXIS == 1) ? M_GROUP : M_EFF;
constexpr uint32_t STAGE_COLS = (SPLIT_AXIS == 0) ? HN_BLOCK : HN_PAD;
constexpr uint32_t STAGE_TILES = STAGE_ROWS * STAGE_COLS;

void kernel_main() {
    const uint32_t wu_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t kr = get_arg_val<uint32_t>(2);
    const uint32_t kstart = get_arg_val<uint32_t>(3);
    const uint32_t is_root = get_arg_val<uint32_t>(4);
    const uint32_t parent_x = get_arg_val<uint32_t>(5);
    const uint32_t parent_y = get_arg_val<uint32_t>(6);

    const auto wu_acc = TensorAccessor(wu_args, wu_addr, BFP4_TILE);
    const auto out_acc = TensorAccessor(out_args, out_addr, BFP8_TILE);

    {
        MaybeDeviceZoneScope("writer_prep");
        // W_up: NoC1 twin of the reader's W_gate load — same coalescing (or lack of it) knobs.
        if constexpr (SPLIT_AXIS == 0) {
            for (uint32_t s = 0; s < S; ++s) {
                const uint32_t wucb = CB_WU_BASE + s;
                cb_reserve_back(wucb, KR_PAD * STAGE_COLS);
                const uint32_t wp = get_write_ptr(wucb);
                for (uint32_t k = 0; k < kr; ++k) {
                    for (uint32_t n = 0; n < STAGE_COLS; ++n) {
                        noc_async_read(
                            wu_acc.get_noc_addr((kstart + k) * HN_PAD + s * STAGE_COLS + n),
                            wp + (k * STAGE_COLS + n) * BFP4_TILE,
                            BFP4_TILE);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(wucb, KR_PAD * STAGE_COLS);
            }
        } else {
            cb_reserve_back(CB_WU_BASE, KR_PAD * HN_PAD);
            const uint32_t wp = get_write_ptr(CB_WU_BASE);
            for (uint32_t k = 0; k < kr; ++k) {
                for (uint32_t n = 0; n < HN_PAD; ++n) {
                    noc_async_read(
                        wu_acc.get_noc_addr((kstart + k) * HN_PAD + n), wp + (k * HN_PAD + n) * BFP4_TILE, BFP4_TILE);
                }
            }
            noc_async_read_barrier();
            cb_push_back(CB_WU_BASE, KR_PAD * HN_PAD);
        }
    }

    if (!is_root) {
        MaybeDeviceZoneScope("writer_reduce_child");
        volatile tt_l1_ptr uint32_t* sem_go_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uint32_t>(get_semaphore(SEM_GO)));
        const uint32_t sem_data_addr = static_cast<uint32_t>(get_semaphore(SEM_DATA));
        uint32_t invites = 0;
        for (uint32_t s = 0; s < S; ++s) {
            const uint32_t ab = CB_A_BASE + s;
            const uint32_t bb = CB_B_BASE + s;
            cb_wait_front(ab, STAGE_TILES);
            cb_wait_front(bb, STAGE_TILES);
            // Parent invites us once per stage; SEM_GO is monotone, no reset needed (real op).
            noc_semaphore_wait_min(sem_go_ptr, ++invites);
            noc_async_write(
                get_read_ptr(ab),
                get_noc_addr(parent_x, parent_y, get_write_ptr(CB_REDUCE_GATE_BASE + s)),
                STAGE_TILES * BFP8_TILE);
            noc_async_write(
                get_read_ptr(bb),
                get_noc_addr(parent_x, parent_y, get_write_ptr(CB_REDUCE_UP_BASE + s)),
                STAGE_TILES * BFP8_TILE);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(parent_x, parent_y, sem_data_addr), 1);
            cb_pop_front(ab, STAGE_TILES);
            cb_pop_front(bb, STAGE_TILES);
        }
    } else {
        // Root: write each stage's final SwiGLU slice (CB_B) to its DRAM sub-region.
        //   HN-split: stage s owns ALL M_EFF rows, columns [s*HN_BLOCK, (s+1)*HN_BLOCK).
        //   M-split:  stage s owns rows [s*M_GROUP, (s+1)*M_GROUP), ALL HN_PAD columns.
        MaybeDeviceZoneScope("writer_out");
        for (uint32_t s = 0; s < S; ++s) {
            const uint32_t bb = CB_B_BASE + s;
            cb_wait_front(bb, STAGE_TILES);
            const uint32_t rp = get_read_ptr(bb);
            const uint32_t row_base = (SPLIT_AXIS == 1) ? s * STAGE_ROWS : 0;
            const uint32_t col_base = (SPLIT_AXIS == 0) ? s * STAGE_COLS : 0;
            for (uint32_t m = 0; m < STAGE_ROWS; ++m) {
                for (uint32_t n = 0; n < STAGE_COLS; ++n) {
                    noc_async_write(
                        rp + (m * STAGE_COLS + n) * BFP8_TILE,
                        out_acc.get_noc_addr((row_base + m) * HN_PAD + col_base + n),
                        BFP8_TILE);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(bb, STAGE_TILES);
        }
    }
}
