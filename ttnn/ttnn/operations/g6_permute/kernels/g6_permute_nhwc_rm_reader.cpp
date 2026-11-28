// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_permute reader (nhwc_rm_transpose, L1 input). Unit (n, h, wb) = input pages
// [(n * H + h) * W + wb * block_w, + width). Page k of a unit lands in slot k of slot_bytes (one row of
// a ct-tile tilize block), so each CB page of 32 slots is one w-tile of the unit. A unit always takes
// kw pages (a tail unit pushes pad pages it never reads) so it never straddles the CB wrap: compute
// addresses all its w-tiles from one read pointer. Slot rows past the unit's width stay stale; they
// only reach output columns the writer never sends.
// Each w-tile's reads carry transaction id 1 or 2, and a w-tile is pushed only after the next one's
// reads are issued (its barrier waits on its own id), so issue never stalls on read latency and the
// first push is one w-tile. The write address is tracked here because the pending w-tile is still
// unpushed when the next one is reserved.
// Raw dataflow: no kernel_lib helper reads ROW_MAJOR pages into strided CB slots.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_input_slots = get_compile_time_arg_val(0);
constexpr uint32_t slot_bytes = get_compile_time_arg_val(1);
constexpr uint32_t in_page_bytes = get_compile_time_arg_val(2);
constexpr uint32_t kw = get_compile_time_arg_val(3);
constexpr uint32_t units_per_row = get_compile_time_arg_val(4);
constexpr uint32_t block_w = get_compile_time_arg_val(5);
constexpr uint32_t tail_w = get_compile_time_arg_val(6);
constexpr uint32_t cb_pages = get_compile_time_arg_val(7);
constexpr auto input_args = TensorAccessorArgs<8>();

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t page_bytes = TILE_ROWS * slot_bytes;
static_assert(cb_pages % kw == 0 && cb_pages >= kw);
static_assert(slot_bytes >= in_page_bytes);

template <typename Accessor>
FORCE_INLINE void issue_pages(const Accessor& input, uint32_t& page, uint32_t slot, uint32_t count) {
    for (uint32_t p = 0; p < count; ++p) {
        noc_async_read<in_page_bytes>(input.get_noc_addr(page), slot, in_page_bytes);
        slot += slot_bytes;
        ++page;
    }
}

struct ReadState {
    uint32_t cb_start, cb_end, unit_addr, pending, trid;
};

// Reserves room for the pending w-tiles plus this unit, issues each w-tile under the free transaction id
// and pushes the previous one once its reads have landed. Issue and barrier are sum zones (~5 cycles a
// step against ~51 for a scoped zone, and no marker cap): they cover every unit of the core.
template <typename Accessor>
FORCE_INLINE void load_unit(const Accessor& input, ReadState& s, uint32_t& page, uint32_t width) {
    const uint32_t kwu = (width + TILE_ROWS - 1) / TILE_ROWS;
    cb_reserve_back(cb_input_slots, s.pending + kw);
    const uint32_t base = s.unit_addr;
    s.unit_addr += kw * page_bytes;
    if (s.unit_addr == s.cb_end) {
        s.unit_addr = s.cb_start;
    }
    for (uint32_t wt = 0; wt < kwu; ++wt) {
        const uint32_t p_begin = wt * TILE_ROWS;
        const uint32_t p_end = (wt + 1) * TILE_ROWS < width ? (wt + 1) * TILE_ROWS : width;
        noc_async_read_set_trid(s.trid);
        {
            DeviceZoneScopedSumN1("PERM-RD-ISSUE");
            issue_pages(input, page, base + p_begin * slot_bytes, p_end - p_begin);
        }
        if (s.pending > 0) {
            {
                DeviceZoneScopedSumN2("PERM-RD-BARRIER");
                noc_async_read_barrier_with_trid(3 - s.trid);
            }
            cb_push_back(cb_input_slots, s.pending);
        }
        s.pending = 1;
        s.trid = 3 - s.trid;
    }
    s.pending += kw - kwu;
}

void kernel_main() {
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    uint32_t page = get_arg_val<uint32_t>(0);
    uint32_t remaining = get_arg_val<uint32_t>(1);
    uint32_t row_pos = get_arg_val<uint32_t>(2);

    const auto input = TensorAccessor(input_args, input_addr, in_page_bytes);
    const uint32_t cb_start = get_write_ptr(cb_input_slots);
    ReadState s{cb_start, cb_start + cb_pages * page_bytes, cb_start, 0, 1};

    // The first unit alone is a scoped zone: it shows the pipeline fill.
    for (bool first = true; remaining > 0; --remaining, first = false) {
        const uint32_t width = row_pos == units_per_row - 1 ? tail_w : block_w;
        if (first) {
            DeviceZoneScopedN("PERM-RD-FIRST-UNIT");
            load_unit(input, s, page, width);
        } else {
            load_unit(input, s, page, width);
        }
        if (++row_pos == units_per_row) {
            row_pos = 0;
        }
    }
    if (s.pending > 0) {
        DeviceZoneScopedSumN2("PERM-RD-BARRIER");
        noc_async_read_barrier_with_trid(3 - s.trid);
        cb_push_back(cb_input_slots, s.pending);
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier();
}
