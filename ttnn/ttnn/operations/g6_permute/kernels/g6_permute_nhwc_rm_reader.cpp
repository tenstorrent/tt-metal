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
// Page issue is a bank cursor on raw command-buffer registers: one division per w-tile gives the first
// page's (bank, bank address); each page then steps the bank (and the bank address at the wrap) and takes
// its NoC XY and bank offset from the firmware's bank tables. A page costs four register writes and a
// ready poll: ~23 cycles against ~38 for noc_async_read + TensorAccessor::get_noc_addr (p44); see
// cursor_issue for where the plain read stays.
// Raw dataflow: no kernel_lib helper reads ROW_MAJOR pages into strided CB slots, and the one-packet
// set_state/with_state reads cannot change the NoC coordinate per page.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_input_slots = get_compile_time_arg_val(0);
constexpr uint32_t slot_bytes = get_compile_time_arg_val(1);
constexpr uint32_t in_page_bytes = get_compile_time_arg_val(2);
constexpr uint32_t kw = get_compile_time_arg_val(3);
constexpr uint32_t units_per_row = get_compile_time_arg_val(4);
constexpr uint32_t block_w = get_compile_time_arg_val(5);
constexpr uint32_t tail_w = get_compile_time_arg_val(6);
constexpr uint32_t cb_pages = get_compile_time_arg_val(7);
// The input's bank stride (aligned page size) and bank count.
constexpr uint32_t bank_stride = get_compile_time_arg_val(8);
constexpr uint32_t num_banks = get_compile_time_arg_val(9);

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t page_bytes = TILE_ROWS * slot_bytes;
static_assert(cb_pages % kw == 0 && cb_pages >= kw);
static_assert(slot_bytes >= in_page_bytes && bank_stride >= in_page_bytes);
static_assert(num_banks == NUM_L1_BANKS, "L1-interleaved input: host bank count = firmware bank count");
static_assert(noc_mode == DM_DEDICATED_NOC);

// Pages above 128 B keep noc_async_read: at the cursor's issue rate 110 cores would ask the NoC for
// ~1.2 KB/cycle, past what it delivers, and delivered throughput collapses (C = 128, 256 B pages: +31 %).
constexpr bool cursor_issue = in_page_bytes <= 128;

// Pages [page, page + count) into consecutive slots from `slot`, on read_cmd_buf (its transaction id, set
// by noc_async_read_set_trid, is kept).
FORCE_INLINE void issue_pages(uint32_t input_addr, uint32_t& page, uint32_t slot, uint32_t count) {
    if constexpr (!cursor_issue) {
        const InterleavedAddrGen<false> gen{.bank_base_address = input_addr, .page_size = bank_stride};
        for (uint32_t p = 0; p < count; ++p) {
            noc_async_read<in_page_bytes>(gen.get_noc_addr(page), slot, in_page_bytes);
            slot += slot_bytes;
            ++page;
        }
        return;
    }
    constexpr uint32_t noc = noc_index;
    constexpr uint32_t buf = read_cmd_buf;
    const uint32_t round = page / num_banks;
    uint32_t bank = page - round * num_banks;
    uint32_t src_round = input_addr + round * bank_stride;
    while (!noc_cmd_buf_ready(noc, buf));
    NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_AT_LEN_BE, in_page_bytes);
    for (uint32_t p = 0; p < count; ++p) {
        const uint32_t xy = l1_bank_to_noc_xy[noc][bank];
        const uint32_t src = src_round + bank_to_l1_offset[bank];
        while (!noc_cmd_buf_ready(noc, buf));
        NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_RET_ADDR_LO, slot);
        NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_TARG_ADDR_LO, src);
        NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_TARG_ADDR_COORDINATE, xy);
        NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        slot += slot_bytes;
        if (++bank == num_banks) {
            bank = 0;
            src_round += bank_stride;
        }
    }
    noc_reads_num_issued[noc] += count;
    page += count;
}

struct ReadState {
    uint32_t cb_start, cb_end, unit_addr, pending, trid;
};

// Reserves room for the pending w-tiles plus this unit, issues each w-tile under the free transaction id
// and pushes the previous one once its reads have landed. Issue and barrier are sum zones (~5 cycles a
// step against ~51 for a scoped zone, and no marker cap): they cover every unit of the core.
FORCE_INLINE void load_unit(uint32_t input_addr, ReadState& s, uint32_t& page, uint32_t width) {
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
            issue_pages(input_addr, page, base + p_begin * slot_bytes, p_end - p_begin);
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

    const uint32_t cb_start = get_write_ptr(cb_input_slots);
    ReadState s{cb_start, cb_start + cb_pages * page_bytes, cb_start, 0, 1};

    // The first unit alone is a scoped zone: it shows the pipeline fill.
    for (bool first = true; remaining > 0; --remaining, first = false) {
        const uint32_t width = row_pos == units_per_row - 1 ? tail_w : block_w;
        if (first) {
            DeviceZoneScopedN("PERM-RD-FIRST-UNIT");
            load_unit(input_addr, s, page, width);
        } else {
            load_unit(input_addr, s, page, width);
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
