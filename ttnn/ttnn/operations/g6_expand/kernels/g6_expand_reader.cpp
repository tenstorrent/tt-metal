// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand reader (NCRISC, NoC0) — regime R1 of op_design.md.
//
// For every input page (column) that this core's column-major output range touches:
//   fetch_row0_block : land row 0 of faces 0 and 1 (row_fetch_bytes each, at offsets 0 and
//                      face_bytes) straight into the column's slot of cb_bcast_tiles; all fetches
//                      of a block are in flight under ONE read barrier.
//   replicate_row0   : in place, copy face0 row0 -> face0 rows 1..15 + face2 rows 0..15 and
//                      face1 row0 -> face1 rows 1..15 + face3 rows 0..15 (RISC-V word copies;
//                      bytes only, bit-exact; overwrites any fetched padding rows).
//   publish          : push one page per column.
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
    constexpr auto input_args = TensorAccessorArgs<7>();

    constexpr uint32_t face_rows = face_bytes / face_row_bytes;

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_t = get_arg_val<uint32_t>(0);
    const uint32_t n_t = get_arg_val<uint32_t>(1);
    if (n_t == 0) {
        return;
    }

    const auto input_accessor = TensorAccessor(input_args, input_addr, page_size);

    const uint32_t first_col = start_t / Ht;
    const uint32_t last_col = (start_t + n_t - 1) / Ht;
    const uint32_t cols_this_core = last_col - first_col + 1;

    for (uint32_t c0 = 0; c0 < cols_this_core; c0 += col_block) {
        const uint32_t block_cols = (cols_this_core - c0) < col_block ? (cols_this_core - c0) : col_block;
        cb_reserve_back(cb_bcast_tiles, block_cols);
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
        for (uint32_t j = 0; j < block_cols; ++j) {
            {
                DeviceZoneScopedN("G6E_REPLICATE");
                const uint32_t slot = block_base + j * page_size;
                // face0 row0 -> face0 rows 1..15, then faces 2 rows 0..15 (contiguous after face 1)
                fill_rows<face_row_bytes>(slot, slot + face_row_bytes, face_rows - 1);
                fill_rows<face_row_bytes>(slot, slot + 2 * face_bytes, face_rows);
                // face1 row0 -> face1 rows 1..15, face3 rows 0..15
                fill_rows<face_row_bytes>(slot + face_bytes, slot + face_bytes + face_row_bytes, face_rows - 1);
                fill_rows<face_row_bytes>(slot + face_bytes, slot + 3 * face_bytes, face_rows);
            }
            cb_push_back(cb_bcast_tiles, 1);
        }
    }
}
