// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ckv_bfp4_rows reader (deepseek_v41_flash_d_p MB.4): this core's bands of 32 rows of a ROW_MAJOR bf16 row tensor (one
// row a page, interleaved or ND-sharded), each band's 32 rows read in order into cb 0 as 16 tile-sized pages (the
// row-major layout tilize_block expects). The cores split nb bands as ckv_share does (CKV_ROWS_NC cores, row-wise
// index over a CKV_ROWS_GX-wide grid).
// compile-time args: row bytes, src accessor
// common args: src addr, src_row0 (per-run value), nb (per-run value: bands of the whole call); no per-core args

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ckv_share.hpp"

void kernel_main() {
    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t src_row0 = get_common_arg_val<uint32_t>(1);
    const uint32_t nb = get_common_arg_val<uint32_t>(2);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();
    const auto src = TensorAccessor(src_args, src_addr, row_bytes);
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t TILES = 16;

    uint32_t band0, n_bands;
    ckv_share(nb / CKV_ROWS_NC, nb % CKV_ROWS_NC, CKV_ROWS_GX, band0, n_bands);
    for (uint32_t b = band0; b < band0 + n_bands; ++b) {
        cb_reserve_back(cb_in, TILES);
        uint32_t l1 = get_write_ptr(cb_in);
        const uint32_t r0 = src_row0 + b * 32;
        for (uint32_t r = 0; r < 32; ++r, l1 += row_bytes) {
            noc_async_read(src.get_noc_addr(r0 + r), l1, row_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, TILES);
    }
}
