// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// read_inflight bake-off reader — the ONE stage under test.
//
// Every arm reads the SAME contiguous run of row-major sticks
// (`num_blocks * TILE_H` sticks starting at `start_stick`, `row_bytes` each,
// taken at `col_off` bytes into the stick) into the same tile-page CB, and
// pushes them in the same order. The ONLY thing that varies is HOW MANY BYTES
// ARE IN FLIGHT PER READ BARRIER:
//
//   VARIANT_HELPER (0) — the op's CURRENT approach, verbatim: one call to
//       dataflow_kernel_lib::read_sticks_for_tilize<TILE>. That helper barriers
//       once per TILE-ROW (tilize_helpers_dataflow.inl ~line 126), i.e. NT_BLK
//       is structurally pinned to 1: TILE_H sticks in flight, never more.
//
//   VARIANT_RAW (1) — HELPER SUBSTITUTION (capability, not ergonomics): the
//       helper's barrier cadence is INTERNAL and has no parameter, so
//       "NT_BLK tile-rows under one barrier" (op design lamp L3) is
//       INEXPRESSIBLE through it. This arm reproduces the helper's loop body
//       byte-for-byte (same TensorAccessor, same noc_async_read, same L1
//       stride, same CB handshake) and only groups NT_BLK tile-rows per
//       reserve/barrier/push. NT_BLK == 1 makes it the helper's exact schedule,
//       which is what turns "raw vs helper" into a measurable control arm
//       rather than an assumption.
//
//   VARIANT_TRID (2) — master.md B8 double-issue on top of the same grouping:
//       group i's reads are issued BEFORE group i-1's barrier, so a request is
//       always in flight across the group boundary. Two fixed CB slots
//       (CB_DEPTH == 2 exactly), transaction ids alternating 1 <-> 2.
//       The trid is reset to 0 before return — a live packet tag on a command
//       buffer is a firmware-level hang after kernel_main returns.
//
// Deeper CB (master.md C16) is a HOST knob (`cb_depth`), not a variant: it
// changes how far the reader may run ahead of compute, not how it issues.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = 0;

    constexpr uint32_t variant = get_compile_time_arg_val(0);  // 0 helper / 1 raw / 2 raw+trid
    constexpr uint32_t tile_h = get_compile_time_arg_val(1);
    constexpr uint32_t wt_chunk = get_compile_time_arg_val(2);
    constexpr uint32_t nt_blk = get_compile_time_arg_val(3);  // tile-rows per read barrier
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(4);
    // VARIANT_AHEAD only: how many GROUPS may be outstanding before the reader
    // blocks on the oldest one's barrier. 1 reproduces VARIANT_TRID exactly.
    constexpr uint32_t ahead = get_compile_time_arg_val(5);
    constexpr uint32_t cb_depth = get_compile_time_arg_val(6);
    constexpr auto src_args = TensorAccessorArgs<7>();

    constexpr uint32_t TILE_W = 32;
    constexpr uint32_t row_bytes = wt_chunk * TILE_W * elem_bytes;
    constexpr uint32_t grp_pages = nt_blk * wt_chunk;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_stick = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t col_off = get_arg_val<uint32_t>(3);

    if (num_blocks == 0) {
        return;
    }
    const auto acc = TensorAccessor(src_args, src_addr);

    if constexpr (variant == 0) {
        // ── BASELINE: the op's current approach, unmodified ───────────────
        dataflow_kernel_lib::read_sticks_for_tilize<cb_in, dataflow_kernel_lib::TilizeGranularity::TILE>(
            acc,
            /*total_num_rows=*/num_blocks * tile_h,
            /*row_bytes=*/row_bytes,
            /*start_page=*/start_stick,
            /*byte_offset_within_page=*/col_off);
    } else if constexpr (variant == 1) {
        // ── NT_BLK grouping ───────────────────────────────────────────────
        uint32_t stick = start_stick;
        uint32_t left = num_blocks;
        while (left > 0) {
            const uint32_t nb = left < nt_blk ? left : nt_blk;
            const uint32_t rows = nb * tile_h;
            cb_reserve_back(cb_in, nb * wt_chunk);
            uint32_t l1 = get_write_ptr(cb_in);
            for (uint32_t r = 0; r < rows; ++r) {
                noc_async_read(acc.get_noc_addr(stick + r, col_off), l1, row_bytes);
                l1 += row_bytes;  // aligned path: the L1 stride IS row_bytes
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, nb * wt_chunk);
            stick += rows;
            left -= nb;
        }
    } else if constexpr (variant == 3) {
        // ── NT_BLK grouping + N-DEEP issue-ahead (generalised B8) ─────────
        // The two-slot B8 form can only ever hold ONE group in flight, and with
        // a depth-2 CB it must additionally wait for the CB to drain completely
        // before issuing the next group — so it buys in-flight depth by giving
        // up pipeline slack. This arm decouples the two: `ahead` groups may be
        // outstanding, the CB is `cb_depth >= ahead + 1` groups deep, and the
        // PUSH granularity stays one group, so compute is never made to wait for
        // a bigger read to land.
        constexpr uint32_t slot_bytes = grp_pages * get_tile_size(cb_in);
        constexpr uint32_t n_trid = ahead + 1;
        const uint32_t slot_base = get_write_ptr(cb_in);
        uint32_t slot = 0;
        uint32_t trid_issue = 1, trid_wait = 1;
        uint32_t pending = 0;
        uint32_t stick = start_stick;
        uint32_t left = num_blocks;
        while (left > 0) {
            const uint32_t rows = nt_blk * tile_h;
            cb_reserve_back(cb_in, (pending + 1) * grp_pages);
            uint32_t l1 = slot_base + slot * slot_bytes;
            noc_async_read_set_trid(trid_issue);
            for (uint32_t r = 0; r < rows; ++r) {
                noc_async_read(acc.get_noc_addr(stick + r, col_off), l1, row_bytes);
                l1 += row_bytes;
            }
            slot = (slot + 1 == cb_depth) ? 0 : slot + 1;
            trid_issue = (trid_issue == n_trid) ? 1 : trid_issue + 1;
            ++pending;
            stick += rows;
            left -= nt_blk;

            if (pending > ahead) {
                noc_async_read_barrier_with_trid(trid_wait);
                cb_push_back(cb_in, grp_pages);
                trid_wait = (trid_wait == n_trid) ? 1 : trid_wait + 1;
                --pending;
            }
        }
        while (pending > 0) {
            noc_async_read_barrier_with_trid(trid_wait);
            cb_push_back(cb_in, grp_pages);
            trid_wait = (trid_wait == n_trid) ? 1 : trid_wait + 1;
            --pending;
        }
        // MANDATORY: leave the command buffer's packet tag clear.
        noc_async_read_set_trid(0);
    } else {
        // ── NT_BLK grouping + B8 double-issue ─────────────────────────────
        // The host only builds this arm with CB_DEPTH == 2 and nt_blk dividing
        // num_blocks, so the write pointer alternates between two fixed slots
        // and every group is full size (no wrap arithmetic needed).
        constexpr uint32_t slot_bytes = grp_pages * get_tile_size(cb_in);
        const uint32_t slot_base = get_write_ptr(cb_in);
        uint32_t slot = 0;
        uint32_t trid_issue = 1, trid_wait = 1;
        bool in_flight = false;
        uint32_t stick = start_stick;
        uint32_t left = num_blocks;
        while (left > 0) {
            const uint32_t nb = left < nt_blk ? left : nt_blk;
            const uint32_t rows = nb * tile_h;
            // Room for the still-unpushed in-flight group AND this one.
            cb_reserve_back(cb_in, in_flight ? 2 * grp_pages : grp_pages);
            uint32_t l1 = slot_base + slot * slot_bytes;
            noc_async_read_set_trid(trid_issue);
            for (uint32_t r = 0; r < rows; ++r) {
                noc_async_read(acc.get_noc_addr(stick + r, col_off), l1, row_bytes);
                l1 += row_bytes;
            }
            slot ^= 1;
            trid_issue ^= 3;  // alternate 1 <-> 2

            if (in_flight) {
                noc_async_read_barrier_with_trid(trid_wait);
                cb_push_back(cb_in, grp_pages);
                trid_wait ^= 3;
            }
            in_flight = true;
            stick += rows;
            left -= nb;
        }
        noc_async_read_barrier_with_trid(trid_wait);  // drain the last group
        cb_push_back(cb_in, grp_pages);
        // MANDATORY: leave the command buffer's packet tag clear.
        noc_async_read_set_trid(0);
    }
}
