// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post reader (NCRISC, NoC0) — COLUMN-STREAM candidate (perf_experiments/column_stream).
//
// The CB handoff granularity is a GROUP of G columns (host CT arg; G = 1 is pure column streaming), decoupled
// from the number of bytes in flight: up to INFLIGHT groups are issued ahead, each tagged with its own NoC
// transaction id (trid = 1 + group seq % 15), and pushed IN ORDER as soon as that trid's reads have landed, while
// the later groups stay in flight. Per group: G F tiles (slot c) and n*G X tiles (slot i*G + c), valid
// columns only, nominal G / n*G pushes (so every push is G-aligned and never straddles the CB wrap; the host
// sizes both CBs to a multiple of G).
//
// Slots of issued-but-unpushed groups are claimed by reserving (m + 1) groups past the push pointer
// (m = groups in flight) and addressing write_ptr + m * G pages (wrapped once at fifo_limit): the CB counts only
// pushed pages, so this is the standard reserve contract extended to m unpublished groups.
//
// Raw NoC API (bypasses nothing in kernel_lib: there is no dataflow helper for trid-pipelined CB fills):
// noc_async_read_set_trid / ncrisc_noc_read_with_transaction_id_flushed / noc_async_read_barrier_with_trid.
// COEF_EXPANDER must be "writer" (the reader issues data reads only; the host asserts it).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "mhc_post_common.hpp"

void kernel_main() {
    // ---- compile-time args ----
    constexpr uint32_t n = get_compile_time_arg_val(0);                  // streams
    constexpr uint32_t col_tiles_per_row = get_compile_time_arg_val(1);  // Ct = C / 32
    constexpr uint32_t group_col_tiles = get_compile_time_arg_val(2);    // G
    constexpr uint32_t sublayer_page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t residual_page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sublayer_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t cb_residual_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t inflight_groups = get_compile_time_arg_val(7);  // groups issued ahead of the push pointer
    constexpr uint32_t batch_col_tiles = get_compile_time_arg_val(8);  // RB: columns issued together (multiple of G)
    constexpr uint32_t tail_cols = get_compile_time_arg_val(9);        // trailing columns issued one group at a time
    constexpr auto sublayer_args = TensorAccessorArgs<10>();
    constexpr auto residual_args = TensorAccessorArgs<sublayer_args.next_compile_time_args_offset()>();

    // trids 1..15: trid 0 is the untagged default (other agents' reads on this NoC may count against it).
    constexpr uint32_t NUM_TRIDS = NOC_MAX_TRANSACTION_ID;  // 15
    static_assert(inflight_groups >= 1 && inflight_groups <= NUM_TRIDS, "one trid per in-flight group");
    auto trid_of = [](uint32_t seq) { return 1 + seq % NUM_TRIDS; };
    static_assert(batch_col_tiles % group_col_tiles == 0, "a read batch is whole groups");
    static_assert(batch_col_tiles / group_col_tiles <= inflight_groups, "a read batch fits the in-flight window");
    constexpr uint32_t residual_row_tiles = n * col_tiles_per_row;
    constexpr uint32_t residual_group_tiles = n * group_col_tiles;

    // ---- runtime args ----
    const uint32_t sublayer_addr = get_arg_val<uint32_t>(0);
    const uint32_t residual_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_unit = get_arg_val<uint32_t>(2);
    const uint32_t num_units = get_arg_val<uint32_t>(3);

    const auto sublayer_acc = TensorAccessor(sublayer_args, sublayer_addr, sublayer_page_bytes);
    const auto residual_acc = TensorAccessor(residual_args, residual_addr, residual_page_bytes);

    uint32_t issued = 0;  // groups whose reads were issued
    uint32_t pushed = 0;  // groups pushed (always <= issued; in flight = issued - pushed)

    auto push_oldest = [&]() {
        cb_push_back(cb_sublayer_tiles, group_col_tiles);
        cb_push_back(cb_residual_tiles, residual_group_tiles);
        ++pushed;
    };
    // Publish every leading group whose reads have all landed (in order; non-blocking).
    auto push_landed = [&]() {
        while (pushed != issued && ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid_of(pushed))) {
            push_oldest();
        }
    };
    auto wait_push_oldest = [&]() {
        noc_async_read_barrier_with_trid(trid_of(pushed));
        push_oldest();
    };
    // Byte address of the slot `pages` past the CB's write pointer (one wrap at most: pages < CB capacity).
    auto slot_addr = [](uint32_t cb, uint32_t pages, uint32_t page_bytes) {
        uint32_t a = get_write_ptr(cb) + pages * page_bytes;
        if (a >= get_local_cb_interface(cb).fifo_limit) {
            a -= get_local_cb_interface(cb).fifo_size;
        }
        return a;
    };

    uint32_t units_left = num_units;
    mhc_post::SegmentWalker walker(start_unit, num_units, col_tiles_per_row);
    while (!walker.done()) {
        const mhc_post::Segment seg = walker.next();
        for (uint32_t done = 0; done < seg.col_tiles;) {
            // Batch width: RB columns, or one group once the core's range is within its last tail_cols columns
            // (fine-grained tail: columns land, and are pushed, progressively instead of all at the batch's end).
            const uint32_t width = units_left <= tail_cols ? group_col_tiles : batch_col_tiles;
            const uint32_t bvalid = seg.col_tiles - done < width ? seg.col_tiles - done : width;
            const uint32_t col_start = seg.col0 + done;
            done += bvalid;
            units_left -= bvalid;
            const uint32_t ng = mhc_post::num_blocks(bvalid, group_col_tiles);  // groups in this batch

            push_landed();
            if (issued - pushed + ng > inflight_groups) {
                MaybeDeviceZoneScope("reader_barrier");  // in-flight window full: wait the oldest group(s)
                do {
                    wait_push_oldest();
                } while (issued - pushed + ng > inflight_groups);
            }
            // Space for the in-flight groups plus this batch (pushing does not change the requirement; compute pops
            // do).
            {
                uint32_t m = issued - pushed;
                if (!cb_pages_reservable_at_back(cb_sublayer_tiles, (m + ng) * group_col_tiles) ||
                    !cb_pages_reservable_at_back(cb_residual_tiles, (m + ng) * residual_group_tiles)) {
                    MaybeDeviceZoneScope("reader_reserve");  // back-pressure from compute
                    do {
#ifndef CS_RESERVE_NOPOLL
                        push_landed();  // keep compute fed while we wait on it
#endif
                        m = issued - pushed;
                    } while (!cb_pages_reservable_at_back(cb_sublayer_tiles, (m + ng) * group_col_tiles) ||
                             !cb_pages_reservable_at_back(cb_residual_tiles, (m + ng) * residual_group_tiles));
                }
            }
            MaybeDeviceZoneScope("reader_issue");
            const uint32_t m = issued - pushed;
            // Term-major over the batch (F of every column, then X_0 of every column, ...): the op's request order,
            // which rotates DRAM banks per request; each read carries its column group's trid.
            for (uint32_t t = 0; t <= n; ++t) {
                for (uint32_t gi = 0; gi < ng; ++gi) {
                    const uint32_t c0 = gi * group_col_tiles;
                    const uint32_t gvalid = bvalid - c0 < group_col_tiles ? bvalid - c0 : group_col_tiles;
                    noc_async_read_set_trid(trid_of(issued + gi));
#ifndef CS_STUB_DM
                    if (t == 0) {
                        const uint32_t f_base =
                            slot_addr(cb_sublayer_tiles, (m + gi) * group_col_tiles, sublayer_page_bytes);
                        const uint32_t f_page0 = seg.row * col_tiles_per_row + col_start + c0;
                        for (uint32_t c = 0; c < gvalid; ++c) {
                            noc_async_read(
                                sublayer_acc.get_noc_addr(f_page0 + c),
                                f_base + c * sublayer_page_bytes,
                                sublayer_page_bytes);
                        }
                    } else {
                        const uint32_t i = t - 1;
                        const uint32_t x_slot0 =
                            slot_addr(cb_residual_tiles, (m + gi) * residual_group_tiles, residual_page_bytes) +
                            i * group_col_tiles * residual_page_bytes;
                        const uint32_t x_page0 = seg.row * residual_row_tiles + i * col_tiles_per_row + col_start + c0;
                        for (uint32_t c = 0; c < gvalid; ++c) {
                            noc_async_read(
                                residual_acc.get_noc_addr(x_page0 + c),
                                x_slot0 + c * residual_page_bytes,
                                residual_page_bytes);
                        }
                    }
#else
                    (void)gvalid;
#endif
                }
            }
            // The last request must have left the command buffer (and be counted against its trid) before a
            // later non-blocking flushed() check may read "0 outstanding" for these groups.
            while (!noc_cmd_buf_ready(noc_index, read_cmd_buf)) {
            }
            issued += ng;
        }
    }
    {
        MaybeDeviceZoneScope("reader_barrier");  // drain: the last groups (tail of the read stream)
        while (pushed != issued) {
            wait_push_oldest();
        }
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier();
}
