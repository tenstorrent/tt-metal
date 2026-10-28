// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Per band of 32 rows (16 bfloat4_b tiles across 512 columns): each tile's bytes scattered into the band's 32 row
// records in L1 (row = 32 exponent bytes, one per 16-value group, then 256 bytes of nibbles, value 2i in the low nibble
// of byte i), then the 32 rows written as pages row0 + 32 band + i of the row table. With CKV_ROUNDED_OUT each tile's
// bf16 copy (the same rounded values) goes to the rounded output at its tile id.
// bfloat4_b tile bytes: 64 exponent bytes (face 0..3 x 16 rows), then 4 faces x 16 rows x 8 bytes.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t rows_addr = get_arg_val<uint32_t>(0);
    const uint32_t band0 = get_arg_val<uint32_t>(1);
    const uint32_t n_bands = get_arg_val<uint32_t>(2);
    const uint32_t row0 = get_arg_val<uint32_t>(3);
#ifdef CKV_ROUNDED_OUT
    const uint32_t rounded_addr = get_arg_val<uint32_t>(4);
#endif

    constexpr uint32_t cb_bfp4 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rows = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rounded = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_band = get_compile_time_arg_val(3);  // 16 (512 columns)
    constexpr uint32_t row_bytes = get_compile_time_arg_val(4);       // 288
    constexpr auto rows_args = TensorAccessorArgs<5>();
    const auto rows_acc = TensorAccessor(rows_args, rows_addr, row_bytes);
#ifdef CKV_ROUNDED_OUT
    constexpr auto rounded_args = TensorAccessorArgs<rows_args.next_compile_time_args_offset()>();
    const uint32_t rounded_tile_bytes = get_tile_size(cb_rounded);
    const auto rounded_acc = TensorAccessor(rounded_args, rounded_addr, rounded_tile_bytes);
#endif
    constexpr uint32_t n_groups = tiles_per_band * 2;

    cb_reserve_back(cb_rows, 1);  // the band's 32 rows, a scratch buffer (never pushed)
    const uint32_t rows_l1 = get_write_ptr(cb_rows);

    for (uint32_t b = band0; b < band0 + n_bands; b++) {
        for (uint32_t c = 0; c < tiles_per_band; c++) {
            cb_wait_front(cb_bfp4, 1);
            const uint32_t tile = get_read_ptr(cb_bfp4);
            volatile tt_l1_ptr uint8_t* tb = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tile);
            for (uint32_t r = 0; r < 32; r++) {
                const uint32_t fr = r & 15;
                const uint32_t face0 = (r >> 4) * 2;
                const uint32_t row = rows_l1 + r * row_bytes;
                volatile tt_l1_ptr uint8_t* rb = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(row);
                rb[2 * c] = tb[face0 * 16 + fr];
                rb[2 * c + 1] = tb[(face0 + 1) * 16 + fr];
                volatile tt_l1_ptr uint32_t* src0 =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile + 64 + face0 * 128 + fr * 8);
                volatile tt_l1_ptr uint32_t* src1 =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile + 64 + (face0 + 1) * 128 + fr * 8);
                volatile tt_l1_ptr uint32_t* dst =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row + n_groups + 16 * c);
                dst[0] = src0[0];
                dst[1] = src0[1];
                dst[2] = src1[0];
                dst[3] = src1[1];
            }
            cb_pop_front(cb_bfp4, 1);
#ifdef CKV_ROUNDED_OUT
            cb_wait_front(cb_rounded, 1);
            noc_async_write(
                get_read_ptr(cb_rounded), rounded_acc.get_noc_addr(b * tiles_per_band + c), rounded_tile_bytes);
            noc_async_writes_flushed();
            cb_pop_front(cb_rounded, 1);
#endif
        }
        for (uint32_t r = 0; r < 32; r++) {
            noc_async_write(rows_l1 + r * row_bytes, rows_acc.get_noc_addr(row0 + b * 32 + r), row_bytes);
        }
        noc_async_write_barrier();  // the scratch rows are rewritten by the next band
    }
}
