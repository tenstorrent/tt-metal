// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_permute writer (blocks_rm_gather, blocks_rm_bank_coalesced): gather_block + write_block.
// out[n, c, i, h, j, w] = x[n, c, h, w, i, j]. Per unit (n, c, h) the reader's slots hold the
// [W, I, J] values (page (w, i) = J values, page_stride apart); the gather moves them into I * J
// output rows of W values in cb_output_rows (g6_permute_rm_transpose.hpp), and each row goes to
// output page p_out = ((n * C + c) * I + i) * H * J + h * J + j.
// cb_output_rows is this kernel's private staging area: it is never pushed or popped.

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
constexpr auto output_args = TensorAccessorArgs<10>();

constexpr uint32_t unit_slot_bytes = tensor_w * tensor_i * page_stride;
constexpr uint32_t unit_row_bytes = tensor_i * tensor_j * out_row_bytes;
// Output pages of consecutive i of one unit are this far apart.
constexpr uint32_t i_page_stride = tensor_h * tensor_j;

static_assert(out_page_bytes == tensor_w * sizeof(uint16_t), "output page is one row of W bfloat16 values");
static_assert(out_row_bytes >= out_page_bytes);

// Per i, the [W, J] values of slots (w, i) become rows (i, j).
FORCE_INLINE void gather_unit(uint32_t src_unit, uint32_t dst_unit) {
    for (uint32_t i = 0; i < tensor_i; ++i) {
        g6_permute::transpose_pages<tensor_w, tensor_j, tensor_i * page_stride, out_row_bytes>(
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

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    uint32_t remaining = get_arg_val<uint32_t>(0);
    OutCursor cursor{get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2)};

    const auto output = TensorAccessor(output_args, output_addr, out_page_bytes);
    const uint32_t rows_base = get_write_ptr(cb_output_rows);

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

    // The next program starts right after this one: no write may still be in flight.
    {
        DeviceZoneScopedN("PERM-WR-FINAL-BARRIER");
        noc_async_write_barrier();
    }
}
