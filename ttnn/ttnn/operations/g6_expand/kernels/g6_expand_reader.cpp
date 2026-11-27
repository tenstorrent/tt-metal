// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand reader (NCRISC, NoC0) — regime R1 of op_design.md.
//
// For every input page (column) that this core's column-major output range touches:
//   fetch_row0_block : land row 0 of faces 0 and 1 (row_fetch_bytes each, at offsets 0 and
//                      face_bytes) straight into the column's slot of cb_bcast_tiles; all fetches
//                      of a block are in flight under ONE read barrier.
//   replicate_row0   : in place, face0 row0 -> face0 rows 1..15 + face2 rows 0..15 and
//                      face1 row0 -> face1 rows 1..15 + face3 rows 0..15 (bytes only, bit-exact;
//                      overwrites any fetched padding rows). Mechanism knob `cpu_rows` (CT):
//                        cpu_rows == 2*face_rows : all rows by RISC-V word copies, per column;
//                        cpu_rows <  face_rows   : RISC-V builds rows [0, cpu_rows) of faces 0/1, then
//                          local NoC L1->L1 doubling copies grow them to 16 rows, then one local
//                          copy of faces 0/1 onto faces 2/3 — every level batched over the block's
//                          columns under one barrier (perf lamp "replication mechanism").
//   publish          : push one page per column.
//   write_share      : (knob reader_write_eighths, CT) the reader also writes the last
//                      (seg_len * reader_write_eighths) / 8 tiles of every column segment on NoC0,
//                      straight from the slot it built (perf lamp "write issue on one RISC"); the
//                      writer derives the same split. Writes are flushed before the slot could be
//                      re-reserved and barriered before return (BB-03).
// No helper applies: the op is pure byte movement (compute-path broadcast would change bits; R5).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {

template <uint32_t face_row_bytes>
FORCE_INLINE void fill_rows(uint32_t src_row_addr, uint32_t dst_addr, uint32_t num_rows) {
    constexpr uint32_t words = face_row_bytes / sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_row_addr);
    uint32_t row[words];
#pragma GCC unroll 8
    for (uint32_t w = 0; w < words; ++w) {
        row[w] = src[w];
    }
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    for (uint32_t r = 0; r < num_rows; ++r) {
#pragma GCC unroll 8
        for (uint32_t w = 0; w < words; ++w) {
            dst[w] = row[w];
        }
        dst += words;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_bcast_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t col_block = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t row_fetch_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t cpu_rows = get_compile_time_arg_val(7);
    constexpr uint32_t reader_write_eighths = get_compile_time_arg_val(8);
    constexpr uint32_t Wt = get_compile_time_arg_val(9);
    constexpr auto input_args = TensorAccessorArgs<10>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    constexpr uint32_t face_rows = face_bytes / face_row_bytes;
    constexpr bool all_cpu = cpu_rows >= 2 * face_rows;
    static_assert(all_cpu || (cpu_rows >= 1 && cpu_rows <= face_rows && (cpu_rows & (cpu_rows - 1)) == 0));

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t output_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t start_t = get_arg_val<uint32_t>(0);
    const uint32_t n_t = get_arg_val<uint32_t>(1);
    if (n_t == 0) {
        return;
    }

    const auto input_accessor = TensorAccessor(input_args, input_addr, page_size);
    [[maybe_unused]] const auto output_accessor = TensorAccessor(output_args, output_addr, page_size);
    const uint32_t end_t = start_t + n_t;

    const uint32_t first_col = start_t / Ht;
    const uint32_t last_col = (start_t + n_t - 1) / Ht;
    const uint32_t cols_this_core = last_col - first_col + 1;

    for (uint32_t c0 = 0; c0 < cols_this_core; c0 += col_block) {
        const uint32_t block_cols = (cols_this_core - c0) < col_block ? (cols_this_core - c0) : col_block;
        {
            DeviceZoneScopedN("G6E_RESERVE");  // blocking wait: own zone (BB-21)
            cb_reserve_back(cb_bcast_tiles, block_cols);
        }
        const uint32_t block_base = get_write_ptr(cb_bcast_tiles);
        {
            DeviceZoneScopedN("G6E_FETCH");
            for (uint32_t j = 0; j < block_cols; ++j) {
                const uint32_t col = first_col + c0 + j;
                const uint32_t slot = block_base + j * page_size;
                noc_async_read(input_accessor.get_noc_addr(col, 0), slot, row_fetch_bytes);
                noc_async_read(input_accessor.get_noc_addr(col, face_bytes), slot + face_bytes, row_fetch_bytes);
            }
            noc_async_read_barrier();
        }
        if constexpr (all_cpu) {
            for (uint32_t j = 0; j < block_cols; ++j) {
                {
                    DeviceZoneScopedN("G6E_REPLICATE");
                    const uint32_t slot = block_base + j * page_size;
                    // face0 row0 -> face0 rows 1..15, face2 rows 0..15
                    fill_rows<face_row_bytes>(slot, slot + face_row_bytes, face_rows - 1);
                    fill_rows<face_row_bytes>(slot, slot + 2 * face_bytes, face_rows);
                    // face1 row0 -> face1 rows 1..15, face3 rows 0..15
                    fill_rows<face_row_bytes>(slot + face_bytes, slot + face_bytes + face_row_bytes, face_rows - 1);
                    fill_rows<face_row_bytes>(slot + face_bytes, slot + 3 * face_bytes, face_rows);
                }
                cb_push_back(cb_bcast_tiles, 1);
            }
        } else {
            {
                DeviceZoneScopedN("G6E_REPLICATE");
                for (uint32_t j = 0; j < block_cols; ++j) {
                    const uint32_t slot = block_base + j * page_size;
                    if constexpr (cpu_rows > 1) {
                        fill_rows<face_row_bytes>(slot, slot + face_row_bytes, cpu_rows - 1);
                        fill_rows<face_row_bytes>(slot + face_bytes, slot + face_bytes + face_row_bytes, cpu_rows - 1);
                    }
                }
                for (uint32_t rows = cpu_rows; rows < face_rows; rows *= 2) {
                    const uint32_t bytes = rows * face_row_bytes;
                    for (uint32_t j = 0; j < block_cols; ++j) {
                        const uint32_t slot = block_base + j * page_size;
                        noc_async_read(get_noc_addr(slot), slot + bytes, bytes);
                        noc_async_read(get_noc_addr(slot + face_bytes), slot + face_bytes + bytes, bytes);
                    }
                    noc_async_read_barrier();
                }
                for (uint32_t j = 0; j < block_cols; ++j) {
                    const uint32_t slot = block_base + j * page_size;
                    noc_async_read(get_noc_addr(slot), slot + 2 * face_bytes, 2 * face_bytes);
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb_bcast_tiles, block_cols);
        }
        if constexpr (reader_write_eighths > 0) {
            DeviceZoneScopedN("G6E_RD_WRITE");
            for (uint32_t j = 0; j < block_cols; ++j) {
                const uint32_t col = first_col + c0 + j;
                const uint32_t seg_begin = start_t > col * Ht ? start_t : col * Ht;
                const uint32_t seg_end = end_t < (col + 1) * Ht ? end_t : (col + 1) * Ht;
                const uint32_t mine = ((seg_end - seg_begin) * reader_write_eighths) / 8;
                const uint32_t img = col / Wt;
                const uint32_t wt = col - img * Wt;
                const uint32_t slot = block_base + j * page_size;
                uint32_t page = (img * Ht + (seg_end - mine - col * Ht)) * Wt + wt;
                for (uint32_t k = 0; k < mine; ++k, page += Wt) {
                    noc_async_write(slot, output_accessor.get_noc_addr(page), page_size);
                }
            }
            noc_async_writes_flushed();  // slot may be re-reserved by the next block
        }
    }
    if constexpr (reader_write_eighths > 0) {
        DeviceZoneScopedN("G6E_RD_DRAIN");
        noc_async_write_barrier();
    }
}
