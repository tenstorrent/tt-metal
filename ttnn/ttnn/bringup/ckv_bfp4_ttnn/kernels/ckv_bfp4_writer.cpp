// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Per band of 32 rows: the band's 32 row records (packed by the packer into one cb_rows page, 320 B apart) written
// as pages row0 + 32 band + i of the row table; with CKV_ROUNDED_OUT the band's 16 bf16 tiles to the rounded output.
// Only NoC writes: the records are assembled by the packer (ckv_bfp4_compute.cpp).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ckv_share.hpp"

void kernel_main() {
#ifdef CKV_COMMON_GX  // (option compact: common [rows, per, extra, row0[, rounded]], no per-core args)
    const uint32_t rows_addr = get_common_arg_val<uint32_t>(0);
    uint32_t band0, n_bands;
    ckv_share(get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2), CKV_COMMON_GX, band0, n_bands);
    const uint32_t row0 = get_common_arg_val<uint32_t>(3);
#ifdef CKV_ROUNDED_OUT
    const uint32_t rounded_addr = get_common_arg_val<uint32_t>(4);
#endif
#else
    const uint32_t rows_addr = get_arg_val<uint32_t>(0);
    const uint32_t band0 = get_arg_val<uint32_t>(1);
    const uint32_t n_bands = get_arg_val<uint32_t>(2);
    const uint32_t row0 = get_arg_val<uint32_t>(3);
#ifdef CKV_ROUNDED_OUT
    const uint32_t rounded_addr = get_arg_val<uint32_t>(4);
#endif
#endif

    constexpr uint32_t cb_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rounded = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_band = get_compile_time_arg_val(2);  // 16 (512 columns)
    constexpr uint32_t row_bytes = get_compile_time_arg_val(3);       // 320
    constexpr auto rows_args = TensorAccessorArgs<4>();
    const auto rows_acc = TensorAccessor(rows_args, rows_addr, row_bytes);
#ifdef CKV_ROUNDED_OUT
    constexpr auto rounded_args = TensorAccessorArgs<rows_args.next_compile_time_args_offset()>();
    const uint32_t rounded_tile_bytes = get_tile_size(cb_rounded);
    const auto rounded_acc = TensorAccessor(rounded_args, rounded_addr, rounded_tile_bytes);
#endif

    for (uint32_t b = band0; b < band0 + n_bands; b++) {
#ifdef CKV_ROUNDED_OUT
        cb_wait_front(cb_rounded, tiles_per_band);
        uint32_t t_l1 = get_read_ptr(cb_rounded);
        for (uint32_t c = 0; c < tiles_per_band; c++, t_l1 += rounded_tile_bytes) {
            noc_async_write(t_l1, rounded_acc.get_noc_addr(b * tiles_per_band + c), rounded_tile_bytes);
        }
#endif
        cb_wait_front(cb_rows, 1);
        const uint32_t rows_l1 = get_read_ptr(cb_rows);
#ifdef CKV_BLOCKS
        noc_async_write(rows_l1, rows_acc.get_noc_addr(row0 / 32 + b), row_bytes);  // one page a band
#else
        for (uint32_t r = 0; r < 32; r++) {
            noc_async_write(rows_l1 + r * row_bytes, rows_acc.get_noc_addr(row0 + b * 32 + r), row_bytes);
        }
#endif
        noc_async_write_barrier();
        cb_pop_front(cb_rows, 1);
#ifdef CKV_ROUNDED_OUT
        cb_pop_front(cb_rounded, tiles_per_band);
#endif
    }
}
