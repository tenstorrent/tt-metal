// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_permute writer (blocks_rm_gather, blocks_rm_bank_coalesced): gather_block + write_block.
// out[n, c, i, h, j, w] = x[n, c, h, w, i, j]. Per unit (n, c, h) the reader's slots hold the
// [W, I, J] values (page (w, i) = J values, page_stride apart); the gather moves them into I * J
// output rows of W values in cb_output_rows (g6_permute_rm_transpose.hpp), and each row goes to
// output page p_out = ((n * C + c) * I + i) * H * J + h * J + j.
// cb_output_rows is this kernel's private staging area: it is never pushed or popped.
// Grouped writes (write_group_units > 0, DRAM output): for a fixed (n, c, i), consecutive h give
// consecutive output pages, so the pages of one bank are consecutive slots of that bank, as in the
// coalesced read. Up to write_group_units consecutive units of one (n, c) form a write group, whose
// rows the gather places bank by bank: run i (pages first_i + t, t < units * J) has one region per
// bank, page first_i + t in region (b0_i + t) % num_banks at slot (b0_i + t) / num_banks, with
// b0_i = first_i % num_banks and slots out_slot_bytes apart like the bank's pages. Each (run, bank)
// is then one write of whole slots. The J rows of one (unit, i) sit one region apart: J divides
// num_banks and first_i + t is a multiple of J, so they never wrap past the last bank.
// write_group_depth buffers alternate; each buffer's writes carry their own transaction id, so
// the gather into one buffer only waits for that buffer's previous writes to leave L1.

#include "api/dataflow/dataflow_api.h"
#include "g6_permute_common.hpp"
#include "g6_permute_rm_transpose.hpp"

constexpr uint32_t cb_input_slots = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_rows = get_compile_time_arg_val(1);
constexpr uint32_t block_h = get_compile_time_arg_val(2);
constexpr uint32_t tensor_h = get_compile_time_arg_val(3);
constexpr uint32_t tensor_w = get_compile_time_arg_val(4);
constexpr uint32_t tensor_i = get_compile_time_arg_val(5);
constexpr uint32_t tensor_j = get_compile_time_arg_val(6);
// Byte distance of consecutive input pages in cb_input_slots.
constexpr uint32_t page_stride = get_compile_time_arg_val(7);
constexpr uint32_t out_page_bytes = get_compile_time_arg_val(8);
constexpr uint32_t out_row_bytes = get_compile_time_arg_val(9);
// Page pairs whose loads the gather issues together (g6_permute::transpose_pages).
constexpr uint32_t gather_pair_batch = get_compile_time_arg_val(10);
// 0: one write per output row.
constexpr uint32_t write_group_units = get_compile_time_arg_val(11);
constexpr uint32_t write_group_depth = get_compile_time_arg_val(12);
constexpr uint32_t num_banks = get_compile_time_arg_val(13);
// Bank stride of the output pages (its aligned page size).
constexpr uint32_t out_slot_bytes = get_compile_time_arg_val(14);
// Slots of one bank region of a write group.
constexpr uint32_t group_slots = get_compile_time_arg_val(15);
constexpr auto output_args = TensorAccessorArgs<16>();

constexpr uint32_t unit_slot_bytes = tensor_w * tensor_i * page_stride;
constexpr uint32_t unit_row_bytes = tensor_i * tensor_j * out_row_bytes;
// Output pages of consecutive i of one unit are this far apart.
constexpr uint32_t i_page_stride = tensor_h * tensor_j;

static_assert(out_page_bytes == tensor_w * sizeof(uint16_t), "output page is one row of W bfloat16 values");
static_assert(out_row_bytes >= out_page_bytes);

constexpr bool grouped_writes = write_group_units != 0;
constexpr uint32_t region_bytes = group_slots * out_slot_bytes;
constexpr uint32_t run_bytes = num_banks * region_bytes;
constexpr uint32_t group_bytes = tensor_i * run_bytes;
// Transaction ids of the group buffers' writes.
constexpr uint32_t FIRST_GROUP_TRID = 1;

static_assert(!grouped_writes || num_banks % tensor_j == 0, "a unit's J rows stay in consecutive banks");
static_assert(!grouped_writes || (write_group_depth >= 1 && FIRST_GROUP_TRID + write_group_depth <= 16));
static_assert(!grouped_writes || (out_slot_bytes >= out_page_bytes && out_slot_bytes % 16 == 0));
static_assert(!grouped_writes || region_bytes <= NOC_MAX_BURST_SIZE, "one (run, bank) write is one packet");

// Per i, the [W, J] values of slots (w, i) become rows (i, j).
FORCE_INLINE void gather_unit(uint32_t src_unit, uint32_t dst_unit) {
    for (uint32_t i = 0; i < tensor_i; ++i) {
        g6_permute::transpose_pages<tensor_w, tensor_j, tensor_i * page_stride, out_row_bytes, gather_pair_batch>(
            src_unit + i * page_stride, dst_unit + i * tensor_j * out_row_bytes);
    }
}

FORCE_INLINE void gather_block(uint32_t rows_base, uint32_t extent) {
    uint32_t src = get_read_ptr(cb_input_slots);
    uint32_t dst = rows_base;
    // Plain loads and stores: keep them after the flush that made the slots current and before the
    // NoC writes that read the rows.
    asm volatile("" ::: "memory");
    for (uint32_t u = 0; u < extent; ++u) {
        gather_unit(src, dst);
        src += unit_slot_bytes;
        dst += unit_row_bytes;
    }
    asm volatile("" ::: "memory");
}

// Output cursor of the unit being written: its h and p_out(n, c, i = 0, h, j = 0).
struct OutCursor {
    uint32_t h, page;

    void advance() {
        page += tensor_j;
        if (++h == tensor_h) {
            h = 0;
            page += (tensor_i - 1) * i_page_stride;
        }
    }
};

template <typename Accessor>
FORCE_INLINE void write_block(const Accessor& output, uint32_t rows_base, OutCursor& cursor, uint32_t extent) {
    uint32_t row = rows_base;
    for (uint32_t u = 0; u < extent; ++u) {
        for (uint32_t i = 0; i < tensor_i; ++i) {
            const uint32_t page = cursor.page + i * i_page_stride;
            for (uint32_t j = 0; j < tensor_j; ++j) {
                noc_async_write<out_page_bytes>(row, output.get_noc_addr(page + j), out_page_bytes);
                row += out_row_bytes;
            }
        }
        cursor.advance();
    }
}

// The write group being gathered: its buffer, the output page of its first unit at i = 0 and j = 0,
// and its units so far.
struct WriteGroup {
    uint32_t buffer, base, first_page, units;
};

// Row (unit, i, j = 0) of the group: its unit offset t = units * J in run i.
FORCE_INLINE uint32_t group_row(const WriteGroup& group, uint32_t i) {
    const uint32_t q = (group.first_page + i * i_page_stride) % num_banks + group.units * tensor_j;
    return group.base + i * run_bytes + (q % num_banks) * region_bytes + (q / num_banks) * out_slot_bytes;
}

// One write of whole slots per (run, bank) of the group, under the group buffer's transaction id.
template <typename Accessor>
FORCE_INLINE void write_group(const Accessor& output, const WriteGroup& group) {
    const uint32_t run_pages = group.units * tensor_j;
    const uint32_t banks = run_pages < num_banks ? run_pages : num_banks;
    // The rows are plain stores; the NoC reads them from L1.
    asm volatile("" ::: "memory");
    for (uint32_t i = 0; i < tensor_i; ++i) {
        const uint32_t first = group.first_page + i * i_page_stride;
        const uint32_t b0 = first % num_banks;
        const uint32_t run = group.base + i * run_bytes;
        for (uint32_t k = 0; k < banks; ++k) {
            const uint32_t q = b0 + k;
            const uint32_t count = (run_pages - k + num_banks - 1) / num_banks;
            noc_async_write_one_packet_with_trid(
                run + (q % num_banks) * region_bytes + (q / num_banks) * out_slot_bytes,
                output.get_noc_addr(first + k),
                count * out_slot_bytes,
                FIRST_GROUP_TRID + group.buffer);
        }
    }
}

// Starts a group at the unit at `cursor` in the next buffer, once that buffer's previous writes have
// left L1.
FORCE_INLINE void start_group(WriteGroup& group, const OutCursor& cursor, uint32_t groups_base) {
    group.buffer = group.buffer + 1 == write_group_depth ? 0 : group.buffer + 1;
    group.base = groups_base + group.buffer * group_bytes;
    group.first_page = cursor.page;
    group.units = 0;
    noc_async_write_flushed_with_trid(FIRST_GROUP_TRID + group.buffer);
}

template <typename Accessor>
FORCE_INLINE void gather_block_grouped(
    const Accessor& output, WriteGroup& group, OutCursor& cursor, uint32_t extent, uint32_t groups_base) {
    uint32_t src = get_read_ptr(cb_input_slots);
    asm volatile("" ::: "memory");
    for (uint32_t u = 0; u < extent; ++u) {
        if (group.units == write_group_units || (cursor.h == 0 && group.units != 0)) {
            write_group(output, group);
            start_group(group, cursor, groups_base);
        }
        for (uint32_t i = 0; i < tensor_i; ++i) {
            g6_permute::transpose_pages<tensor_w, tensor_j, tensor_i * page_stride, region_bytes, gather_pair_batch>(
                src + i * page_stride, group_row(group, i));
        }
        ++group.units;
        src += unit_slot_bytes;
        cursor.advance();
    }
    asm volatile("" ::: "memory");
}

template <bool zoned, typename Accessor>
FORCE_INLINE void gather_write_block_grouped(
    const Accessor& output, WriteGroup& group, OutCursor& cursor, uint32_t extent, uint32_t groups_base) {
    G6_ZONED_STEP(zoned, "PERM-WR-WAIT", cb_wait_front(cb_input_slots, block_h));
    // Drops stale data-cache lines of the refilled slots.
    invalidate_l1_cache();
    G6_ZONED_STEP(zoned, "PERM-WR-GATHER", gather_block_grouped(output, group, cursor, extent, groups_base));
    cb_pop_front(cb_input_slots, block_h);
}

template <bool zoned, typename Accessor>
FORCE_INLINE void gather_write_block(const Accessor& output, uint32_t rows_base, OutCursor& cursor, uint32_t extent) {
    G6_ZONED_STEP(zoned, "PERM-WR-WAIT", cb_wait_front(cb_input_slots, block_h));
    // The rows are the previous block's write source; they must have left L1 before the gather
    // overwrites them. The flush also drops stale data-cache lines of the refilled slots.
    G6_ZONED_STEP(zoned, "PERM-WR-FLUSH", noc_async_writes_flushed());
    G6_ZONED_STEP(zoned, "PERM-WR-GATHER", gather_block(rows_base, extent));
    cb_pop_front(cb_input_slots, block_h);
    G6_ZONED_STEP(zoned, "PERM-WR-ISSUE", write_block(output, rows_base, cursor, extent));
}

template <typename Accessor>
FORCE_INLINE void gather_write_blocks(
    const Accessor& output, uint32_t rows_base, OutCursor& cursor, uint32_t remaining) {
    uint32_t block_idx = 0;
    for (; block_idx < g6_permute::ZONED_BLOCKS && remaining > 0; ++block_idx) {
        DeviceZoneScopedN("PERM-WR-GATHER-BLOCK");
        const uint32_t extent = remaining < block_h ? remaining : block_h;
        gather_write_block<true>(output, rows_base, cursor, extent);
        remaining -= extent;
    }
    if (remaining > 0) {
        DeviceZoneScopedN("PERM-WR-TAIL");
        while (remaining > 0) {
            const uint32_t extent = remaining < block_h ? remaining : block_h;
            gather_write_block<false>(output, rows_base, cursor, extent);
            remaining -= extent;
        }
    }
}

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    uint32_t remaining = get_arg_val<uint32_t>(0);
    OutCursor cursor{get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2)};

    const auto output = TensorAccessor(output_args, output_addr, out_page_bytes);
    const uint32_t rows_base = get_write_ptr(cb_output_rows);

    if constexpr (grouped_writes) {
        if (remaining > 0) {
            // The first group takes buffer 0; its transaction id has no writes yet.
            WriteGroup group{write_group_depth - 1, 0, 0, 0};
            start_group(group, cursor, rows_base);
            uint32_t block_idx = 0;
            for (; block_idx < g6_permute::ZONED_BLOCKS && remaining > 0; ++block_idx) {
                DeviceZoneScopedN("PERM-WR-GATHER-BLOCK");
                const uint32_t extent = remaining < block_h ? remaining : block_h;
                gather_write_block_grouped<true>(output, group, cursor, extent, rows_base);
                remaining -= extent;
            }
            while (remaining > 0) {
                const uint32_t extent = remaining < block_h ? remaining : block_h;
                gather_write_block_grouped<false>(output, group, cursor, extent, rows_base);
                remaining -= extent;
            }
            write_group(output, group);
        }
    } else {
        gather_write_blocks(output, rows_base, cursor, remaining);
    }

    // The next program starts right after this one: no write may still be in flight.
    {
        DeviceZoneScopedN("PERM-WR-FINAL-BARRIER");
        noc_async_write_barrier();
    }
}
