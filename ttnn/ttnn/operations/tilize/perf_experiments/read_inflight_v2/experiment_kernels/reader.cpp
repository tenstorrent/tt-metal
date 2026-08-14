// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// read_inflight_v2 reader — ONE loop for the whole R_ALIGNED regime.
//
// Every arm reads the SAME contiguous run of row-major sticks
// (`num_blocks * TILE_H` sticks starting at `start_stick`, `row_bytes` each,
// taken at `col_off` bytes into the stick) into the same tile-page CB, and
// pushes them in the same order. Only the SCHEDULE and the TRANSACTION SIZE
// vary.
//
//   VARIANT_HELPER (0) — the op's CURRENT approach, verbatim: one call to
//       dataflow_kernel_lib::read_sticks_for_tilize<TILE>. HELPER BASELINE.
//
//   VARIANT_TRID (2) — the op's EXISTING B8 two-slot double-issue
//       (tilize_reader.cpp, `read_trid` branch), reconstructed. Exactly one
//       group in flight, exactly two CB slots, trids alternating 1 <-> 2.
//
//   VARIANT_AHEAD (3) — THE CANDIDATE. `ahead` groups outstanding over
//       `ahead + 1` rotating transaction ids, CB `cb_depth >= ahead + 1` groups
//       deep, push granularity still ONE group. `ahead == 1 && cb_depth == 2`
//       degenerates to VARIANT_TRID's exact schedule (that equivalence is what
//       lets the B8 special case be deleted); `ahead == 0` degenerates to the
//       plain barrier-per-block loop.
//
//   `coalesce` (compile-time, sticks per NoC transfer) is ORTHOGONAL to the
//       schedule: `coalesce == 1` is one transfer per stick (what the helper and
//       the op do today); `coalesce == C` issues ONE transfer covering C
//       consecutive sticks. That is only ADDRESS-CORRECT when the C source
//       sticks are contiguous in the source bank/core AND the block takes the
//       whole stick (col_off == 0 and row_bytes == the source page). The host
//       only builds coalesced arms where it believes that holds — and the
//       bit-exact gate is what decides whether it actually does.
//
// HELPER BYPASS (capability, not ergonomics): read_sticks_for_tilize owns its CB
// handshake AND its barrier internally (tilize_helpers_dataflow.inl:116-126 —
// one plain noc_async_read per row, one plain noc_async_read_barrier per block).
// It exposes NO transaction id, NO issue-ahead depth and NO multi-stick
// transfer, so neither the schedule nor the transaction size under test here is
// reachable through it. The kernel below reproduces its loop body byte-for-byte
// (same TensorAccessor, same L1 stride == row_bytes on the aligned path, same
// page order) and changes only those two things.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = 0;

    constexpr uint32_t variant = get_compile_time_arg_val(0);
    constexpr uint32_t tile_h = get_compile_time_arg_val(1);
    constexpr uint32_t wt_chunk = get_compile_time_arg_val(2);
    constexpr uint32_t nt_blk = get_compile_time_arg_val(3);  // tile-rows per group
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t ahead = get_compile_time_arg_val(5);  // groups outstanding
    constexpr uint32_t cb_depth = get_compile_time_arg_val(6);
    constexpr uint32_t coalesce = get_compile_time_arg_val(7);  // sticks per transfer
    constexpr auto src_args = TensorAccessorArgs<8>();

    constexpr uint32_t TILE_W = 32;
    constexpr uint32_t row_bytes = wt_chunk * TILE_W * elem_bytes;
    constexpr uint32_t grp_pages = nt_blk * wt_chunk;
    constexpr uint32_t grp_rows = nt_blk * tile_h;
    // Sticks per transfer, clamped so a transfer never leaves the group.
    constexpr uint32_t coal = coalesce > grp_rows ? grp_rows : coalesce;
    constexpr uint32_t xfer_bytes = coal * row_bytes;

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
    } else if constexpr (variant == 3) {
        // ── THE ONE LOOP: NT_BLK grouping + N-deep issue-ahead ────────────
        constexpr uint32_t slot_bytes = grp_pages * get_tile_size(cb_in);
        constexpr uint32_t n_trid = ahead + 1;
        const uint32_t slot_base = get_write_ptr(cb_in);
        uint32_t slot = 0;
        uint32_t trid_issue = 1, trid_wait = 1;
        uint32_t pending = 0;
        uint32_t stick = start_stick;
        uint32_t left = num_blocks;
        while (left > 0) {
            cb_reserve_back(cb_in, (pending + 1) * grp_pages);
            uint32_t l1 = slot_base + slot * slot_bytes;
            if constexpr (ahead > 0) {
                noc_async_read_set_trid(trid_issue);
            }
            for (uint32_t r = 0; r < grp_rows; r += coal) {
                noc_async_read(acc.get_noc_addr(stick + r, col_off), l1, xfer_bytes);
                l1 += xfer_bytes;
            }
            slot = (slot + 1 == cb_depth) ? 0 : slot + 1;
            if constexpr (ahead > 0) {
                trid_issue = (trid_issue == n_trid) ? 1 : trid_issue + 1;
            }
            ++pending;
            stick += grp_rows;
            left -= nt_blk;

            if (pending > ahead) {
                if constexpr (ahead > 0) {
                    noc_async_read_barrier_with_trid(trid_wait);
                    trid_wait = (trid_wait == n_trid) ? 1 : trid_wait + 1;
                } else {
                    noc_async_read_barrier();
                }
                cb_push_back(cb_in, grp_pages);
                --pending;
            }
        }
        while (pending > 0) {
            if constexpr (ahead > 0) {
                noc_async_read_barrier_with_trid(trid_wait);
                trid_wait = (trid_wait == n_trid) ? 1 : trid_wait + 1;
            } else {
                noc_async_read_barrier();
            }
            cb_push_back(cb_in, grp_pages);
            --pending;
        }
        if constexpr (ahead > 0) {
            // MANDATORY: leave the command buffer's packet tag clear.
            noc_async_read_set_trid(0);
        }
    } else {
        // ── VARIANT_TRID: the op's EXISTING two-slot B8 double-issue ──────
        constexpr uint32_t slot_bytes = grp_pages * get_tile_size(cb_in);
        const uint32_t slot_base = get_write_ptr(cb_in);
        uint32_t slot = 0;
        uint32_t trid_issue = 1, trid_wait = 1;
        bool in_flight = false;
        uint32_t stick = start_stick;
        uint32_t left = num_blocks;
        while (left > 0) {
            cb_reserve_back(cb_in, in_flight ? 2 * grp_pages : grp_pages);
            uint32_t l1 = slot_base + slot * slot_bytes;
            noc_async_read_set_trid(trid_issue);
            for (uint32_t r = 0; r < grp_rows; r += coal) {
                noc_async_read(acc.get_noc_addr(stick + r, col_off), l1, xfer_bytes);
                l1 += xfer_bytes;
            }
            slot ^= 1;
            trid_issue ^= 3;  // alternate 1 <-> 2

            if (in_flight) {
                noc_async_read_barrier_with_trid(trid_wait);
                cb_push_back(cb_in, grp_pages);
                trid_wait ^= 3;
            }
            in_flight = true;
            stick += grp_rows;
            left -= nt_blk;
        }
        noc_async_read_barrier_with_trid(trid_wait);  // drain the last group
        cb_push_back(cb_in, grp_pages);
        noc_async_read_set_trid(0);
    }
}
