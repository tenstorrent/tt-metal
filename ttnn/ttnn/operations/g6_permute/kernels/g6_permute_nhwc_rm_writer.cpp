// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_permute writer (nhwc_rm_transpose). For unit (n, h, wb) compute pushes one cb_output_rows page per
// c-tile ct: 32 rows c = 32 * ct + r of kwu * 32 values (row stride kwu * 64 B), row c holding
// out[n, c, h, wb * block_w + (0 .. width)]. Row c goes to output page (n * C + c) * H + h at byte
// offset wb * block_w * 2: block_w is a multiple of 8 values unless a unit is a whole row, so the
// offset keeps the 16 B NoC write alignment of the row. Rows c >= C of the last c-tile are never sent.
// Row issue is a bank cursor on raw command-buffer registers: one division per c-tile page gives row
// c0's (bank, bank address); each next channel steps the page by H, i.e. the bank by H mod num_banks and
// the bank address by H div num_banks slots (one more at the wrap), with NoC XY and bank offset from the
// firmware's bank tables. ~27 cycles a row against ~42 for noc_async_write + TensorAccessor (p44); see
// cursor_issue for where the plain write stays.
// Raw dataflow: kernel_lib has no helper that writes strided partial rows to interleaved pages, and the
// one-packet set_state/with_state writes cannot change the NoC coordinate per row.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_output_rows = get_compile_time_arg_val(0);
constexpr uint32_t tensor_c = get_compile_time_arg_val(1);
constexpr uint32_t tensor_ct = get_compile_time_arg_val(2);
constexpr uint32_t tensor_h = get_compile_time_arg_val(3);
constexpr uint32_t units_per_row = get_compile_time_arg_val(4);
constexpr uint32_t block_w = get_compile_time_arg_val(5);
constexpr uint32_t tail_w = get_compile_time_arg_val(6);
constexpr uint32_t kw = get_compile_time_arg_val(7);
constexpr uint32_t kw_tail = get_compile_time_arg_val(8);
constexpr uint32_t out_page_bytes = get_compile_time_arg_val(9);
constexpr bool out_is_dram = get_compile_time_arg_val(10) != 0;
constexpr uint32_t out_num_banks = get_compile_time_arg_val(11);
// The output's bank stride (its aligned page size).
constexpr uint32_t out_slot_bytes = get_compile_time_arg_val(12);

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t value_bytes = sizeof(uint16_t);

static_assert(tail_w >= 1 && tail_w <= block_w);
static_assert(out_page_bytes == ((units_per_row - 1) * block_w + tail_w) * value_bytes, "output page is one row of W");
static_assert(units_per_row == 1 || (block_w * value_bytes) % 16 == 0, "partial rows start 16 B-aligned");

// Output cursor of the unit being written: its position wb in the row, h, and p_out(n, c = 0, h).
struct OutCursor {
    uint32_t wb, h, page;

    bool is_tail() const { return wb == units_per_row - 1; }

    void advance() {
        if (++wb == units_per_row) {
            wb = 0;
            ++page;
            if (++h == tensor_h) {
                h = 0;
                page += (tensor_c - 1) * tensor_h;
            }
        }
    }
};

static_assert(out_num_banks == (out_is_dram ? NUM_DRAM_BANKS : NUM_L1_BANKS), "host bank count = firmware bank count");
static_assert(out_slot_bytes >= out_page_bytes);
// One channel step (+H pages): the bank moves h_bank_step, the bank address h_round_bytes (+ one slot at
// the wrap).
constexpr uint32_t h_bank_step = tensor_h % out_num_banks;
constexpr uint32_t h_round_bytes = (tensor_h / out_num_banks) * out_slot_bytes;

FORCE_INLINE uint32_t out_bank_xy(uint32_t bank) {
    if constexpr (out_is_dram) {
        return dram_bank_to_noc_xy[noc_index][bank];
    } else {
        return l1_bank_to_noc_xy[noc_index][bank];
    }
}

FORCE_INLINE uint32_t out_bank_offset(uint32_t bank) {
    if constexpr (out_is_dram) {
        return bank_to_dram_offset[bank];
    } else {
        return bank_to_l1_offset[bank];
    }
}

// noc_async_write stays where the cheaper issue measured slower: rows above 128 B, where the NoC is asked
// for more than it delivers and throughput collapses (C = 128, 256 B rows), and DRAM output, whose ports
// retire requests at a fixed rate, so faster writes only crowd out the reader's requests ([1,256,512,32]:
// +3.6 %).
constexpr bool cursor_issue = block_w * value_bytes <= 128 && !out_is_dram;

// The rows of c-tile ct of the unit at `cursor`, non-posted on write_cmd_buf, counted like noc_async_write.
FORCE_INLINE void write_rows(uint32_t output_addr, const OutCursor& cursor, uint32_t ct) {
    if constexpr (!cursor_issue) {
        const InterleavedAddrGen<out_is_dram> gen{.bank_base_address = output_addr, .page_size = out_slot_bytes};
        const bool tail = cursor.is_tail();
        const uint32_t bytes = (tail ? tail_w : block_w) * value_bytes;
        const uint32_t row_stride = (tail ? kw_tail : kw) * TILE_ROWS * value_bytes;
        const uint32_t c0 = ct * TILE_ROWS;
        const uint32_t rows = tensor_c - c0 < TILE_ROWS ? tensor_c - c0 : TILE_ROWS;
        uint32_t src = get_read_ptr(cb_output_rows);
        uint32_t page = cursor.page + c0 * tensor_h;
        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_write<kw * TILE_ROWS * value_bytes>(
                src, gen.get_noc_addr(page, cursor.wb * block_w * value_bytes), bytes);
            src += row_stride;
            page += tensor_h;
        }
        return;
    }
    constexpr uint32_t noc = noc_index;
    constexpr uint32_t buf = write_cmd_buf;
    constexpr uint32_t cmd =
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | NOC_CMD_RESP_MARKED;
    const bool tail = cursor.is_tail();
    const uint32_t bytes = (tail ? tail_w : block_w) * value_bytes;
    const uint32_t row_stride = (tail ? kw_tail : kw) * TILE_ROWS * value_bytes;
    const uint32_t c0 = ct * TILE_ROWS;
    const uint32_t rows = tensor_c - c0 < TILE_ROWS ? tensor_c - c0 : TILE_ROWS;
    const uint32_t page = cursor.page + c0 * tensor_h;
    const uint32_t round = page / out_num_banks;
    uint32_t bank = page - round * out_num_banks;
    uint32_t addr = output_addr + round * out_slot_bytes + cursor.wb * block_w * value_bytes;
    uint32_t src = get_read_ptr(cb_output_rows);
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, rows);
        inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, rows);
    }
    while (!noc_cmd_buf_ready(noc, buf));
    NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_CTRL, cmd);
    NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_AT_LEN_BE, bytes);
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t xy = out_bank_xy(bank);
        const uint32_t dst = addr + out_bank_offset(bank);
        while (!noc_cmd_buf_ready(noc, buf));
        NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_TARG_ADDR_LO, src);
        NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_RET_ADDR_LO, dst);
        NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_RET_ADDR_COORDINATE, xy);
        NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        src += row_stride;
        bank += h_bank_step;
        addr += h_round_bytes;
        if (bank >= out_num_banks) {
            bank -= out_num_banks;
            addr += out_slot_bytes;
        }
    }
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        noc_nonposted_writes_num_issued[noc] += rows;
        noc_nonposted_writes_acked[noc] += rows;
    }
}

// Wait and issue are sum zones (~5 cycles a step, no marker cap): they cover every unit of the core.
FORCE_INLINE void write_unit(uint32_t output_addr, OutCursor& cursor) {
    for (uint32_t ct = 0; ct < tensor_ct; ++ct) {
        {
            DeviceZoneScopedSumN1("PERM-WR-WAIT");
            cb_wait_front(cb_output_rows, 1);
        }
        {
            DeviceZoneScopedSumN2("PERM-WR-ISSUE");
            write_rows(output_addr, cursor, ct);
        }
        // The rows are the writes' source: they must have left L1 before compute refills the page.
        noc_async_writes_flushed();
        cb_pop_front(cb_output_rows, 1);
    }
    cursor.advance();
}

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    uint32_t remaining = get_arg_val<uint32_t>(0);
    OutCursor cursor{get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3)};

    // The first unit alone is a scoped zone: it shows the pipeline fill.
    for (bool first = true; remaining > 0; --remaining, first = false) {
        if (first) {
            DeviceZoneScopedN("PERM-WR-FIRST-UNIT");
            write_unit(output_addr, cursor);
        } else {
            write_unit(output_addr, cursor);
        }
    }

    // The next program starts right after this one: no write may still be in flight.
    {
        DeviceZoneScopedN("PERM-WR-FINAL-BARRIER");
        noc_async_write_barrier();
    }
}
