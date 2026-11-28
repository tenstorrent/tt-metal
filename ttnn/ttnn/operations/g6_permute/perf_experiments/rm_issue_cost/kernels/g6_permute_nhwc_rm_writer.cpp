// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// perf experiment rm_issue_cost: copy of kernels/g6_permute_nhwc_rm_writer.cpp with variant guards.
//   ABL_NO_GATHER     stub the RISC-V transpose (ablated floor; output is garbage, timing only).
//   RIC_WRITE_CURSOR  raw-register write issue: bypasses noc_async_write + TensorAccessor::get_noc_addr.
//                     The C partial rows of a unit go to pages p, p + H, ..., p + (C - 1) H. A (bank, bank
//                     address) cursor stepped by H pages replaces the per-row page -> bank division, the NoC XY
//                     and bank offset come from the firmware's bank tables, and NOC_CTRL / NOC_AT_LEN_BE are set
//                     once per unit, so a row costs 4 register writes and one ready poll.
//   RIC_WRITE_2BUF    with RIC_WRITE_CURSOR, dedicated NoC mode: rows alternate between write_cmd_buf and
//                     write_reg_cmd_buf (unused by this kernel, saved at start and restored at the end); the next
//                     buffer's ready poll is loaded before the current buffer's register writes.
//   RIC_UNIT_PUSH     the reader pushes (and this kernel waits for) one unit at a time; block_units only sets
//                     the ring depth. Rows rotate over block_units unit-sized staging slots.
// The rest is the op's writer verbatim.
//
// g6_permute writer (nhwc_rm_transpose): gather_block + write_block.
// out[n, c, h, w] = x[n, h, w, c]. Unit (n, h, wb) covers w in [wb * block_w, + width), width =
// block_w, or tail_w for the last unit of a row. The reader's slots hold its width input pages of C
// values; the gather (g6_permute_rm_transpose.hpp) turns them into C rows of width values in
// cb_output_rows, and row c goes to output page p_out = (n * C + c) * H + h at byte offset
// wb * block_w * 2. block_w is a multiple of 8 values unless a unit is a whole row, so the offset
// keeps the 16 B NoC write alignment of the staging row.
// cb_output_rows is this kernel's private staging area: it is never pushed or popped.

#include "api/dataflow/dataflow_api.h"
#include "g6_permute_common.hpp"
#include "g6_permute_rm_transpose.hpp"

constexpr uint32_t cb_input_slots = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_rows = get_compile_time_arg_val(1);
constexpr uint32_t block_units = get_compile_time_arg_val(2);
constexpr uint32_t tensor_h = get_compile_time_arg_val(3);
constexpr uint32_t tensor_c = get_compile_time_arg_val(4);
constexpr uint32_t units_per_row = get_compile_time_arg_val(5);
constexpr uint32_t block_w = get_compile_time_arg_val(6);
constexpr uint32_t tail_w = get_compile_time_arg_val(7);
constexpr uint32_t slot_bytes = get_compile_time_arg_val(8);
constexpr uint32_t row_bytes = get_compile_time_arg_val(9);
constexpr uint32_t out_page_bytes = get_compile_time_arg_val(10);
constexpr auto output_args = TensorAccessorArgs<11>();

constexpr uint32_t value_bytes = sizeof(uint16_t);
constexpr uint32_t unit_row_bytes = tensor_c * row_bytes;

static_assert(tail_w >= 1 && tail_w <= block_w && row_bytes >= block_w * value_bytes);
static_assert(out_page_bytes == ((units_per_row - 1) * block_w + tail_w) * value_bytes, "output page is one row of W");
static_assert(units_per_row == 1 || (block_w * value_bytes) % 16 == 0, "partial rows start 16 B-aligned");

// Output cursor of the unit being handled: its position wb in the row, h, and p_out(n, c = 0, h).
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

FORCE_INLINE void gather_block(uint32_t src, uint32_t rows_base, OutCursor cursor, uint32_t extent) {
    uint32_t dst = rows_base;
    // Plain loads and stores: keep them after the flush that made the slots current and before the
    // NoC writes that read the rows.
    asm volatile("" ::: "memory");
    for (uint32_t u = 0; u < extent; ++u) {
        if (tail_w != block_w && cursor.is_tail()) {
            g6_permute::transpose_pages<tail_w, tensor_c, slot_bytes, row_bytes>(src, dst);
            src += tail_w * slot_bytes;
        } else {
            g6_permute::transpose_pages<block_w, tensor_c, slot_bytes, row_bytes>(src, dst);
            src += block_w * slot_bytes;
        }
        dst += unit_row_bytes;
        cursor.advance();
    }
    asm volatile("" ::: "memory");
}

#if defined(RIC_WRITE_CURSOR)
static_assert(!output_args.is_sharded, "the bank cursor walks an interleaved output");
constexpr bool out_is_dram = output_args.is_dram;
constexpr uint32_t out_num_banks = out_is_dram ? NUM_DRAM_BANKS : NUM_L1_BANKS;
constexpr uint32_t out_alignment = out_is_dram ? DRAM_ALIGNMENT : L1_ALIGNMENT;
// The output's bank stride (InterleavedAddrGen::aligned_page_size).
constexpr uint32_t out_slot_bytes = (out_page_bytes + out_alignment - 1) / out_alignment * out_alignment;
// One channel step (+H pages) moves the bank by h_bank_step and the bank address by h_round_bytes, plus one
// slot when the bank wraps.
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

FORCE_INLINE void write_block_cursor(uint32_t output_addr, uint32_t rows_base, OutCursor& cursor, uint32_t extent) {
    constexpr uint32_t noc = noc_index;
    constexpr uint32_t buf = write_cmd_buf;
    constexpr uint32_t cmd =
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | NOC_CMD_RESP_MARKED;
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, extent * tensor_c);
        inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, extent * tensor_c);
    }
    uint32_t row = rows_base;
    for (uint32_t u = 0; u < extent; ++u) {
        const uint32_t bytes = (cursor.is_tail() ? tail_w : block_w) * value_bytes;
        const uint32_t round = cursor.page / out_num_banks;
        uint32_t bank = cursor.page - round * out_num_banks;
        uint32_t addr = output_addr + round * out_slot_bytes + cursor.wb * block_w * value_bytes;
        while (!noc_cmd_buf_ready(noc, buf));
        NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_CTRL, cmd);
        NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_AT_LEN_BE, bytes);
        for (uint32_t c = 0; c < tensor_c; ++c) {
            const uint32_t xy = out_bank_xy(bank);
            const uint32_t dst = addr + out_bank_offset(bank);
            while (!noc_cmd_buf_ready(noc, buf));
            NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_TARG_ADDR_LO, row);
            NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_RET_ADDR_LO, dst);
            NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_RET_ADDR_COORDINATE, xy);
            NOC_CMD_BUF_WRITE_REG(noc, buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
            row += row_bytes;
            bank += h_bank_step;
            addr += h_round_bytes;
            if (bank >= out_num_banks) {
                bank -= out_num_banks;
                addr += out_slot_bytes;
            }
        }
        cursor.advance();
    }
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        noc_nonposted_writes_num_issued[noc] += extent * tensor_c;
        noc_nonposted_writes_acked[noc] += extent * tensor_c;
    }
}
#endif

#if defined(RIC_WRITE_2BUF)
constexpr bool wr_2buf = noc_mode == DM_DEDICATED_NOC;
constexpr uint32_t wr_buf_a = write_cmd_buf;
constexpr uint32_t wr_buf_b = write_reg_cmd_buf;
static_assert(!wr_2buf || wr_buf_a != wr_buf_b);

FORCE_INLINE volatile uint32_t* noc_reg(uint32_t buf, uint32_t reg) {
    return reinterpret_cast<volatile uint32_t*>(NOC_CMD_BUF_INSTANCE_OFFSET(noc_index, buf) + reg);
}

NocCmdBufState wr_buf_b_saved;

FORCE_INLINE void wr_2buf_begin() {
    if constexpr (wr_2buf) {
        while (!noc_cmd_buf_ready(noc_index, wr_buf_b));
        noc_cmd_buf_save_state(noc_index, wr_buf_b, &wr_buf_b_saved);
        NOC_CMD_BUF_WRITE_REG(
            noc_index,
            wr_buf_b,
            NOC_TARG_ADDR_COORDINATE,
            NOC_CMD_BUF_READ_REG(noc_index, wr_buf_a, NOC_TARG_ADDR_COORDINATE));
        NOC_CMD_BUF_WRITE_REG(noc_index, wr_buf_b, NOC_TARG_ADDR_MID, 0);
        NOC_CMD_BUF_WRITE_REG(noc_index, wr_buf_b, NOC_RET_ADDR_MID, 0);
        NOC_CMD_BUF_WRITE_REG(noc_index, wr_buf_b, NOC_PACKET_TAG, 0);
    }
}

FORCE_INLINE void wr_2buf_end() {
    if constexpr (wr_2buf) {
        noc_cmd_buf_restore_state(noc_index, wr_buf_b, &wr_buf_b_saved);
    }
}

#define RIC_WR_STEP(x, y)                              \
    {                                                  \
        uint32_t busy = *noc_reg(y, NOC_CMD_CTRL);     \
        const uint32_t dst = addr + off;               \
        *noc_reg(x, NOC_TARG_ADDR_LO) = row;           \
        *noc_reg(x, NOC_RET_ADDR_LO) = dst;            \
        *noc_reg(x, NOC_RET_ADDR_COORDINATE) = xy;     \
        *noc_reg(x, NOC_CMD_CTRL) = NOC_CTRL_SEND_REQ; \
        row += row_bytes;                              \
        bank += h_bank_step;                           \
        addr += h_round_bytes;                         \
        if (bank >= out_num_banks) {                   \
            bank -= out_num_banks;                     \
            addr += out_slot_bytes;                    \
        }                                              \
        xy = out_bank_xy(bank);                        \
        off = out_bank_offset(bank);                   \
        while (busy != NOC_CTRL_STATUS_READY) {        \
            busy = *noc_reg(y, NOC_CMD_CTRL);          \
        }                                              \
    }

FORCE_INLINE void write_block_cursor2(uint32_t output_addr, uint32_t rows_base, OutCursor& cursor, uint32_t extent) {
    constexpr uint32_t cmd =
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | NOC_CMD_RESP_MARKED;
    uint32_t row = rows_base;
    for (uint32_t u = 0; u < extent; ++u) {
        const uint32_t bytes = (cursor.is_tail() ? tail_w : block_w) * value_bytes;
        const uint32_t round = cursor.page / out_num_banks;
        uint32_t bank = cursor.page - round * out_num_banks;
        uint32_t addr = output_addr + round * out_slot_bytes + cursor.wb * block_w * value_bytes;
        uint32_t xy = out_bank_xy(bank);
        uint32_t off = out_bank_offset(bank);
        while (!noc_cmd_buf_ready(noc_index, wr_buf_a));
        while (!noc_cmd_buf_ready(noc_index, wr_buf_b));
        *noc_reg(wr_buf_a, NOC_CTRL) = cmd;
        *noc_reg(wr_buf_b, NOC_CTRL) = cmd;
        *noc_reg(wr_buf_a, NOC_AT_LEN_BE) = bytes;
        *noc_reg(wr_buf_b, NOC_AT_LEN_BE) = bytes;
        uint32_t c = 0;
        for (; c + 2 <= tensor_c; c += 2) {
            RIC_WR_STEP(wr_buf_a, wr_buf_b);
            RIC_WR_STEP(wr_buf_b, wr_buf_a);
        }
        if (c < tensor_c) {
            RIC_WR_STEP(wr_buf_a, wr_buf_b);
        }
        cursor.advance();
    }
    noc_nonposted_writes_num_issued[noc_index] += extent * tensor_c;
    noc_nonposted_writes_acked[noc_index] += extent * tensor_c;
}
#endif

template <typename Accessor>
FORCE_INLINE void write_block(
    const Accessor& output, uint32_t output_addr, uint32_t rows_base, OutCursor& cursor, uint32_t extent) {
#if defined(RIC_WRITE_2BUF)
    if constexpr (wr_2buf) {
        write_block_cursor2(output_addr, rows_base, cursor, extent);
        return;
    }
#endif
#if defined(ABL_NO_WRITES)
    for (uint32_t u = 0; u < extent; ++u) {
        cursor.advance();
    }
#elif defined(RIC_WRITE_CURSOR)
    write_block_cursor(output_addr, rows_base, cursor, extent);
#else
    uint32_t row = rows_base;
    for (uint32_t u = 0; u < extent; ++u) {
        const uint32_t bytes = (cursor.is_tail() ? tail_w : block_w) * value_bytes;
        const uint32_t offset = cursor.wb * block_w * value_bytes;
        for (uint32_t c = 0; c < tensor_c; ++c) {
            noc_async_write<block_w * value_bytes>(row, output.get_noc_addr(cursor.page + c * tensor_h, offset), bytes);
            row += row_bytes;
        }
        cursor.advance();
    }
#endif
}

#if !defined(RIC_UNIT_PUSH)
template <bool zoned, typename Accessor>
FORCE_INLINE void gather_write_block(
    const Accessor& output, uint32_t output_addr, uint32_t rows_base, OutCursor& cursor, uint32_t extent) {
    G6_ZONED_STEP(zoned, "PERM-WR-WAIT", cb_wait_front(cb_input_slots, block_units));
    // The rows are the previous block's write source; they must have left L1 before the gather
    // overwrites them. The flush also drops stale data-cache lines of the refilled slots.
    G6_ZONED_STEP(zoned, "PERM-WR-FLUSH", noc_async_writes_flushed());
#if !defined(ABL_NO_GATHER)
    G6_ZONED_STEP(zoned, "PERM-WR-GATHER", gather_block(get_read_ptr(cb_input_slots), rows_base, cursor, extent));
#endif
    cb_pop_front(cb_input_slots, block_units);
    G6_ZONED_STEP(zoned, "PERM-WR-ISSUE", write_block(output, output_addr, rows_base, cursor, extent));
}
#else
// One unit per handshake. The rows of unit k sit in staging slot k mod block_units; before a slot is reused,
// the writes issued from it (all those issued before the checkpoint taken when it was last filled... i.e. up
// to the end of its own writes) must have left L1.
template <bool zoned, typename Accessor>
FORCE_INLINE void gather_write_unit(
    const Accessor& output, uint32_t output_addr, uint32_t rows, OutCursor& cursor, uint32_t& slot_checkpoint) {
    G6_ZONED_STEP(zoned, "PERM-WR-WAIT", cb_wait_front(cb_input_slots, 1));
    G6_ZONED_STEP(zoned, "PERM-WR-FLUSH", {
        if constexpr (noc_mode == DM_DEDICATED_NOC) {
            while (!noc_nonposted_writes_sent_since(noc_index, slot_checkpoint));
            invalidate_l1_cache();
        } else {
            noc_async_writes_flushed();
        }
    });
#if !defined(ABL_NO_GATHER)
    G6_ZONED_STEP(zoned, "PERM-WR-GATHER", gather_block(get_read_ptr(cb_input_slots), rows, cursor, 1));
#endif
    cb_pop_front(cb_input_slots, 1);
    G6_ZONED_STEP(zoned, "PERM-WR-ISSUE", write_block(output, output_addr, rows, cursor, 1));
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        slot_checkpoint = noc_nonposted_writes_checkpoint(noc_index);
    }
}
#endif

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    uint32_t remaining = get_arg_val<uint32_t>(0);
    OutCursor cursor{get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3)};

    const auto output = TensorAccessor(output_args, output_addr, out_page_bytes);
    const uint32_t rows_base = get_write_ptr(cb_output_rows);
#if defined(RIC_WRITE_2BUF)
    wr_2buf_begin();
#endif

#if !defined(RIC_UNIT_PUSH)
    uint32_t block_idx = 0;
    for (; block_idx < g6_permute::ZONED_BLOCKS && remaining > 0; ++block_idx) {
        DeviceZoneScopedN("PERM-WR-GATHER-BLOCK");
        const uint32_t extent = remaining < block_units ? remaining : block_units;
        gather_write_block<true>(output, output_addr, rows_base, cursor, extent);
        remaining -= extent;
    }
    if (remaining > 0) {
        DeviceZoneScopedN("PERM-WR-TAIL");
        while (remaining > 0) {
            const uint32_t extent = remaining < block_units ? remaining : block_units;
            gather_write_block<false>(output, output_addr, rows_base, cursor, extent);
            remaining -= extent;
        }
    }
#else
    // Checkpoints of the writes issued from each staging slot (0: nothing issued from it yet).
    uint32_t checkpoints[block_units];
    for (uint32_t s = 0; s < block_units; ++s) {
        checkpoints[s] = noc_nonposted_writes_checkpoint(noc_index);
    }
    uint32_t slot = 0;
    uint32_t unit_idx = 0;
    for (; unit_idx < g6_permute::ZONED_BLOCKS && remaining > 0; ++unit_idx) {
        DeviceZoneScopedN("PERM-WR-GATHER-BLOCK");
        gather_write_unit<true>(output, output_addr, rows_base + slot * unit_row_bytes, cursor, checkpoints[slot]);
        slot = slot + 1 == block_units ? 0 : slot + 1;
        --remaining;
    }
    if (remaining > 0) {
        DeviceZoneScopedN("PERM-WR-TAIL");
        while (remaining > 0) {
            gather_write_unit<false>(output, output_addr, rows_base + slot * unit_row_bytes, cursor, checkpoints[slot]);
            slot = slot + 1 == block_units ? 0 : slot + 1;
            --remaining;
        }
    }
#endif

    // The next program starts right after this one: no write may still be in flight.
    {
        DeviceZoneScopedN("PERM-WR-FINAL-BARRIER");
        noc_async_write_barrier();
    }
#if defined(RIC_WRITE_2BUF)
    wr_2buf_end();
#endif
}
