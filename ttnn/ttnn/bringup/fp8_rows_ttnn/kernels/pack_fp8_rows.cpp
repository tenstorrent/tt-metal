// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.pack_fp8_rows (fp8_rows.py): one data-movement kernel (NoC 0). Per row m of the core's share:
//   bytes [0, 128)    <- the e4m3 row m (128 B page of the [.., M, 128] FP8_E4M3 ROW_MAJOR input)
//   bytes [128, 132)  <- the fp32 scale of row m (4 B page of the [.., M, 1] FLOAT32 ROW_MAJOR input), little endian
//   bytes [132, 192)  zero
// A row whose 128 values are all +-0 (a zeroed pad row: per_token_cast_to_fp8 writes the clamp floor 1e-4 / 448 as
// its scale) is written all zero. Rows go through an L1 staging buffer of B rows at a 64-byte aligned stride of 192 B
// (the DRAM read alignment of Blackhole: the scale lands at row offset 128), whose pad bytes are zeroed once.
// compile-time args: cb_stage, B, GX, e4m3 / scale / out accessors; common args: e4m3, scale, out addresses, per, extra
// (core c = y GX + x gets per + (c < extra) rows from c per + min(c, extra): no per-core runtime args).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_stage = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t GX = get_compile_time_arg_val(2);
    constexpr uint32_t VAL_BYTES = 128;
    constexpr uint32_t SCALE_OFF = 128;
    constexpr uint32_t ROW_BYTES = 192;
    constexpr uint32_t ROW_WORDS = ROW_BYTES / 4;
    constexpr auto e_args = TensorAccessorArgs<3>();
    constexpr auto s_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto e4m3 = TensorAccessor(e_args, get_common_arg_val<uint32_t>(0));
    const auto scale = TensorAccessor(s_args, get_common_arg_val<uint32_t>(1));
    const auto out = TensorAccessor(o_args, get_common_arg_val<uint32_t>(2));
    const uint32_t per = get_common_arg_val<uint32_t>(3);
    const uint32_t extra = get_common_arg_val<uint32_t>(4);
    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t count = per + (c < extra ? 1u : 0u);
    const uint32_t first = c * per + (c < extra ? c : extra);
    if (count == 0) {
        return;
    }

    using ptr_t = volatile tt_l1_ptr uint32_t*;
    cb_reserve_back(cb_stage, 1);  // private scratch (never pushed): B rows + 64 B for the alignment
    const uint32_t base = (get_write_ptr(cb_stage) + 63u) & ~63u;
    {
        ptr_t w = reinterpret_cast<ptr_t>(base);
        for (uint32_t i = 0; i < B * ROW_WORDS; ++i) {
            w[i] = 0;
        }
    }
    for (uint32_t m0 = first; m0 < first + count; m0 += B) {
        const uint32_t n = (first + count - m0) < B ? (first + count - m0) : B;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t row = base + i * ROW_BYTES;
            noc_async_read(e4m3.get_noc_addr(m0 + i), row, VAL_BYTES);
            noc_async_read(scale.get_noc_addr(m0 + i), row + SCALE_OFF, 4);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            ptr_t w = reinterpret_cast<ptr_t>(base + i * ROW_BYTES);
            uint32_t any = 0;
            for (uint32_t k = 0; k < VAL_BYTES / 4; ++k) {
                any |= w[k] & 0x7F7F7F7Fu;  // e4m3 +-0 is 0x00 / 0x80
            }
            if (any == 0) {
                for (uint32_t k = 0; k <= VAL_BYTES / 4; ++k) {  // the values and the scale word
                    w[k] = 0;
                }
            }
        }
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write(base + i * ROW_BYTES, out.get_noc_addr(m0 + i), ROW_BYTES);
        }
        noc_async_write_barrier();
    }
}
