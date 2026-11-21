// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_adaln — affine w / b column fetch + multicast, shared by the writer and the reader.
// The fetch runs on the core's NoC1 RISC when S > 1 (perf_experiments/affine_fetch_fix): the writer on the default
// columns, the reader on the NOC_SWAP_EAST_COLS columns. Fetched by the east columns' NoC0 writer, the w / b reads
// queued behind the chip-wide NoC0 x-read storm and landed at ~42 us (2304x3072), stalling the column's compute
// (c_wbp_prep) and, through the all-gather, its whole group. The reader issues the reads before its first x row and
// finishes (barrier, multicast, push) right after that row, so the fetch never delays the x loads.
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {

constexpr uint32_t cb_weight_rows = 10;
constexpr uint32_t cb_bias_rows = 11;

constexpr uint32_t AFFINE_RECEIVER = 0;
constexpr uint32_t AFFINE_INJECTOR = 1;  // fetches and multicasts to the rest of its column
constexpr uint32_t AFFINE_ALONE = 2;     // single-core column: fetches for itself

// Raw dataflow (bypasses ttnn.Mcast / McastArgs): the helper partitions its receivers into EQUAL consecutive groups,
// but a partially-filled grid (num_cores % grid_x != 0, e.g. 1x1x256x1024 -> 32 cores on a 12-wide grid) has columns
// of unequal height, so one channel with a group per column is not expressible. No receiver-ready handshake: the
// landing CBs are full-size (block_width pages), reserved once and written by nothing else in this program, so the
// only synchronization is the data-ready flag (ordered after the data on the same NoC route).
// Runtime args from `arg`: weight_addr, bias_addr, role, then (injector) the column's virtual NoC multicast rectangle
// x0, y0, x1, y1 (already ordered for this NoC) and num_dests.
// Split into issue (reads, no wait) / finish (barrier, multicast, flag, push) / receive (flag wait, push), so a caller
// can overlap the fetch with other work: the reader issues the reads before its first x row and finishes after it.
struct AffineCol {
    uint32_t role, l1_w, l1_b, sem_addr, seg_bytes, tile_bytes;
};

template <uint32_t block_width, uint32_t affine_fetch_bytes, uint32_t sem_id, typename WArgs, typename BArgs>
FORCE_INLINE AffineCol affine_issue(const WArgs& w_args, const BArgs& b_args, uint32_t w_start, uint32_t arg) {
    AffineCol a;
    a.role = get_arg_val<uint32_t>(arg + 2);
    a.tile_bytes = get_tile_size(cb_weight_rows);
    // Row-broadcast reads only row 0 of faces 0 and 1 (byte offsets 0 and face_bytes): one read spans both.
    a.seg_bytes = a.tile_bytes / 4 + affine_fetch_bytes;
    cb_reserve_back(cb_weight_rows, block_width);
    cb_reserve_back(cb_bias_rows, block_width);
    a.l1_w = get_write_ptr(cb_weight_rows);
    a.l1_b = get_write_ptr(cb_bias_rows);
    a.sem_addr = get_semaphore(sem_id);
    if (a.role != AFFINE_RECEIVER) {
        MaybeDeviceZoneScope("aff_issue");
        const auto w_acc = TensorAccessor(w_args, get_arg_val<uint32_t>(arg + 0), a.tile_bytes);
        const auto b_acc = TensorAccessor(b_args, get_arg_val<uint32_t>(arg + 1), a.tile_bytes);
        for (uint32_t c = 0; c < block_width; ++c) {
            noc_async_read(w_acc.get_noc_addr(w_start + c, 0), a.l1_w + c * a.tile_bytes, a.seg_bytes);
        }
        for (uint32_t c = 0; c < block_width; ++c) {
            noc_async_read(b_acc.get_noc_addr(w_start + c, 0), a.l1_b + c * a.tile_bytes, a.seg_bytes);
        }
    }
    return a;
}

template <uint32_t block_width>
FORCE_INLINE void affine_finish(const AffineCol& a, uint32_t arg) {
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a.sem_addr);
    if (a.role == AFFINE_RECEIVER) {
        MaybeDeviceZoneScope("aff_wait");
        noc_semaphore_wait(sem, 1);
    } else {
        {
            MaybeDeviceZoneScope("aff_barrier");
            noc_async_read_barrier();
        }
        if (a.role == AFFINE_INJECTOR) {
            MaybeDeviceZoneScope("aff_mcast");
            const uint32_t x0 = get_arg_val<uint32_t>(arg + 3);
            const uint32_t y0 = get_arg_val<uint32_t>(arg + 4);
            const uint32_t x1 = get_arg_val<uint32_t>(arg + 5);
            const uint32_t y1 = get_arg_val<uint32_t>(arg + 6);
            const uint32_t num_dests = get_arg_val<uint32_t>(arg + 7);
            constexpr bool linked = true;
            // Linked: every data multicast but the last keeps the column path reserved, so the 2*block_width
            // transactions skip the per-multicast path reservation (perf_experiments/affine_fetch). The flag
            // multicast below is unlinked and ordered after the data.
            for (uint32_t c = 0; c < block_width; ++c) {
                const uint32_t off = c * a.tile_bytes;
                noc_async_write_multicast(
                    a.l1_w + off, get_noc_multicast_addr(x0, y0, x1, y1, a.l1_w + off), a.seg_bytes, num_dests, linked);
                noc_async_write_multicast(
                    a.l1_b + off,
                    get_noc_multicast_addr(x0, y0, x1, y1, a.l1_b + off),
                    a.seg_bytes,
                    num_dests,
                    linked && (c + 1 < block_width));
            }
            *sem = 1;
            noc_semaphore_set_multicast(a.sem_addr, get_noc_multicast_addr(x0, y0, x1, y1, a.sem_addr), num_dests);
            noc_async_write_barrier();
        }
    }
    cb_push_back(cb_weight_rows, block_width);
    cb_push_back(cb_bias_rows, block_width);
}

template <uint32_t block_width, uint32_t affine_fetch_bytes, uint32_t sem_id, typename WArgs, typename BArgs>
FORCE_INLINE void affine_column_mcast(const WArgs& w_args, const BArgs& b_args, uint32_t w_start, uint32_t arg) {
    affine_finish<block_width>(
        affine_issue<block_width, affine_fetch_bytes, sem_id>(w_args, b_args, w_start, arg), arg);
}

}  // namespace
