// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.unpack_fp8_rows (fp8_rows.py), reader (NoC 1). Per band of 32 rows [e4m3 (128 B) | fp32 scale (4 B) |
// 60 B]: the 32 rows read into an L1 stage (64 B aligned, 192 B a row), then
//   cb_e: the band's 4 FP8_E4M3 tiles in face order (row i, bytes 16 j .. 16 j + 15 -> tile j / 2, face
//         2 (i / 16) + j % 2, face row i % 16): word copies only
//   cb_s: one FLOAT32 tile, every element of row i = the row's scale (the SFPU multiply's per-row operand)
// compile-time: cb_e, cb_s, cb_stage, GX, rows accessor. common: rows address. per core (value slots, the same on
// every core): src_page0 (the first row's page: batch x rows + row_lo), nbands (UNPACK_COMMON_VALUES: common args 1,
// 2). Core i = y GX + x takes bands [i per + min(i, extra), + per + (i < extra)) of nbands = per x cores + extra.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_e = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(2);
    constexpr uint32_t GX = get_compile_time_arg_val(3);
    constexpr uint32_t NCORES = get_compile_time_arg_val(4);
    constexpr uint32_t ROW_BYTES = 192;
    constexpr auto r_args = TensorAccessorArgs<5>();
    const auto rows = TensorAccessor(r_args, get_common_arg_val<uint32_t>(0), ROW_BYTES);
#ifdef UNPACK_COMMON_VALUES  // (option common_values: the per-run values as common args, no per-core args)
    const uint32_t src_page0 = get_common_arg_val<uint32_t>(1);
    const uint32_t nbands = get_common_arg_val<uint32_t>(2);
#else
    const uint32_t src_page0 = get_arg_val<uint32_t>(0);
    const uint32_t nbands = get_arg_val<uint32_t>(1);
#endif
    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t per = nbands / NCORES, extra = nbands % NCORES;
    const uint32_t count = per + (c < extra ? 1u : 0u);
    const uint32_t first = c * per + (c < extra ? c : extra);
    if (count == 0) {
        return;
    }

    using ptr_t = volatile tt_l1_ptr uint32_t*;
    cb_reserve_back(cb_stage, 1);  // private scratch (never pushed): 32 rows + 64 B for the alignment
    const uint32_t base = (get_write_ptr(cb_stage) + 63u) & ~63u;
    for (uint32_t b = first; b < first + count; ++b) {
        const uint32_t page = src_page0 + b * 32;
        for (uint32_t i = 0; i < 32; ++i) {
            noc_async_read(rows.get_noc_addr(page + i), base + i * ROW_BYTES, ROW_BYTES);
        }
        noc_async_read_barrier();
        cb_reserve_back(cb_e, 4);
        cb_reserve_back(cb_s, 1);
        ptr_t e = reinterpret_cast<ptr_t>(get_write_ptr(cb_e));
        ptr_t s = reinterpret_cast<ptr_t>(get_write_ptr(cb_s));
        for (uint32_t i = 0; i < 32; ++i) {
            ptr_t src = reinterpret_cast<ptr_t>(base + i * ROW_BYTES);
            const uint32_t face0 = (i >> 4) * 2, r16 = i & 15u;
            for (uint32_t j = 0; j < 8; ++j) {  // 16-byte pieces: e4m3 tile j / 2, face face0 + j % 2
                ptr_t d = e + ((j >> 1) * 1024u + (face0 + (j & 1u)) * 256u + r16 * 16u) / 4u;
                d[0] = src[4 * j];
                d[1] = src[4 * j + 1];
                d[2] = src[4 * j + 2];
                d[3] = src[4 * j + 3];
            }
            const uint32_t sc = src[32];                // the fp32 scale (bytes 128..131)
            ptr_t d0 = s + (face0 * 256u + r16 * 16u);  // fp32: one word an element
            ptr_t d1 = s + ((face0 + 1u) * 256u + r16 * 16u);
            for (uint32_t k = 0; k < 16; ++k) {
                d0[k] = sc;
                d1[k] = sc;
            }
        }
        cb_push_back(cb_e, 4);
        cb_push_back(cb_s, 1);
    }
}
