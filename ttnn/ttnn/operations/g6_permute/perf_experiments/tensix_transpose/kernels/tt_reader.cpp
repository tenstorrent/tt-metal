// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tensix_transpose experiment, reader. x [N, H, W, C] ROW_MAJOR, unit (n, h, wb) = input pages
// [(n * H + h) * W + wb * block_w, + width). Page k of a unit lands in slot k of slot_bytes
// (= ceil(C / 32) * 64 B: one row of a tilize block of Ct tiles), so 32 consecutive slots are one
// RM block of 32 rows the compute's unpacker tilizes. A unit always takes kw CB pages (one page = one
// w-tile of 32 slots) so it never straddles the CB wrap: the compute addresses all its w-tiles from
// one read pointer. Pages are pushed push_tiles w-tiles at a time (first_push_tiles for the first
// group, which is what the compute waits for first). Slot rows past the unit's width are left as they
// are: they become output values past the row end, which the writer never sends.
// use_trid: each group's reads carry transaction id 1 or 2, and a group is pushed after the NEXT
// group's reads are issued (barrier on its own id only), so the issue never stalls on read latency;
// only the core's last group waits for its reads. The reader tracks the write address itself because
// a group can still be unpushed when the next unit is reserved.
// Raw dataflow: no kernel_lib helper reads ROW_MAJOR pages into strided CB slots.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_in = get_compile_time_arg_val(0);
constexpr uint32_t slot_bytes = get_compile_time_arg_val(1);
constexpr uint32_t in_page_bytes = get_compile_time_arg_val(2);
constexpr uint32_t kw = get_compile_time_arg_val(3);
constexpr uint32_t units_per_row = get_compile_time_arg_val(4);
constexpr uint32_t block_w = get_compile_time_arg_val(5);
constexpr uint32_t tail_w = get_compile_time_arg_val(6);
constexpr uint32_t push_tiles = get_compile_time_arg_val(7);
constexpr uint32_t first_push_tiles = get_compile_time_arg_val(8);
constexpr bool use_trid = get_compile_time_arg_val(9) != 0;
constexpr uint32_t cb_pages = get_compile_time_arg_val(10);
constexpr auto input_args = TensorAccessorArgs<11>();

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t page_bytes = TILE_ROWS * slot_bytes;
static_assert(cb_pages % kw == 0 && cb_pages >= kw);
static_assert(!use_trid || cb_pages >= 2 * kw, "a pending group and the next unit both need room");

#ifdef TTX_ZONES
#include "api/debug/kernel_profiler.hpp"
#define TTX_ZONE(name) DeviceZoneScopedN(name)
#else
#define TTX_ZONE(name)
#endif

template <typename Accessor>
FORCE_INLINE void issue_pages(const Accessor& input, uint32_t& page, uint32_t slot, uint32_t count) {
    TTX_ZONE("RD-ISSUE");
    for (uint32_t p = 0; p < count; ++p) {
#ifndef TTX_NO_READS
        noc_async_read<in_page_bytes>(input.get_noc_addr(page), slot, in_page_bytes);
#endif
        slot += slot_bytes;
        ++page;
    }
}

void kernel_main() {
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    uint32_t page = get_arg_val<uint32_t>(0);
    uint32_t remaining = get_arg_val<uint32_t>(1);
    uint32_t row_pos = get_arg_val<uint32_t>(2);

    const auto input = TensorAccessor(input_args, input_addr, in_page_bytes);
    const uint32_t cb_start = get_write_ptr(cb_in);
    const uint32_t cb_end = cb_start + cb_pages * page_bytes;
    uint32_t unit_addr = cb_start;
    bool first = true;
    uint32_t pending = 0;  // w-tiles issued but not pushed (use_trid)
    uint32_t trid = 1;

    while (remaining > 0) {
        const uint32_t width = (row_pos == units_per_row - 1) ? tail_w : block_w;
        const uint32_t kwu = (width + TILE_ROWS - 1) / TILE_ROWS;
        {
            TTX_ZONE("RD-RESERVE");
            cb_reserve_back(cb_in, pending + kw);
        }
        const uint32_t base = unit_addr;
        unit_addr += kw * page_bytes;
        if (unit_addr == cb_end) {
            unit_addr = cb_start;
        }
        uint32_t wt = 0;
        while (wt < kwu) {
            uint32_t group = first ? first_push_tiles : push_tiles;
            first = false;
            if (group > kwu - wt) {
                group = kwu - wt;
            }
            const uint32_t p_begin = wt * TILE_ROWS;
            uint32_t p_end = (wt + group) * TILE_ROWS;
            if (p_end > width) {
                p_end = width;
            }
            if constexpr (use_trid) {
                noc_async_read_set_trid(trid);
                issue_pages(input, page, base + p_begin * slot_bytes, p_end - p_begin);
                if (pending > 0) {
                    TTX_ZONE("RD-BARRIER");
                    noc_async_read_barrier_with_trid(3 - trid);
                    cb_push_back(cb_in, pending);
                }
                pending = group;
                trid = 3 - trid;
            } else {
                issue_pages(input, page, base + p_begin * slot_bytes, p_end - p_begin);
                {
                    TTX_ZONE("RD-BARRIER");
                    noc_async_read_barrier();
                }
                cb_push_back(cb_in, group);
            }
            wt += group;
        }
        if (kwu < kw) {
            // Pad pages keep every unit kw pages long; nothing is read into them.
            if constexpr (use_trid) {
                pending += kw - kwu;
            } else {
                cb_push_back(cb_in, kw - kwu);
            }
        }
        if (++row_pos == units_per_row) {
            row_pos = 0;
        }
        --remaining;
    }
    if constexpr (use_trid) {
        if (pending > 0) {
            noc_async_read_barrier_with_trid(3 - trid);
            cb_push_back(cb_in, pending);
        }
        noc_async_read_set_trid(0);
        noc_async_read_barrier();
    }
}
