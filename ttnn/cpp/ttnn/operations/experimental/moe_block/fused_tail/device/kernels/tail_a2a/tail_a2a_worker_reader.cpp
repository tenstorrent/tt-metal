// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TailA2A worker reader (RISCV_1, NoC0): my column slice of the owners' rows (row-major, 32 rows a row tile) and se's
// tiles, row tile by row tile.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0), SE = get_compile_time_arg_val(1);
    constexpr auto y_args = TensorAccessorArgs<2>();
    constexpr auto s_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    const uint32_t y_addr = get_arg_val<uint32_t>(0), s_addr = get_arg_val<uint32_t>(1), c0 = get_arg_val<uint32_t>(2);
    const uint32_t RT = get_arg_val<uint32_t>(3), NTH = get_arg_val<uint32_t>(4);
    const auto y = TensorAccessor(y_args, y_addr, NTH * 64);
    const auto s = TensorAccessor(s_args, s_addr, 2048);
    for (uint32_t r = 0; r < RT; ++r) {
        cb_reserve_back(0, W);
        const uint32_t l1 = get_write_ptr(0);
        for (uint32_t i = 0; i < 32; ++i) {
            noc_async_read(y.get_noc_addr(r * 32 + i, c0 * 64), l1 + i * W * 64, W * 64);
        }
        if constexpr (SE) {
            cb_reserve_back(2, W);
            const uint32_t s1 = get_write_ptr(2);
            for (uint32_t t = 0; t < W; ++t) {
                noc_async_read(s.get_noc_addr(r * NTH + c0 + t), s1 + t * 2048, 2048);
            }
        }
        noc_async_read_barrier();
        cb_push_back(0, W);
        if constexpr (SE) {
            cb_push_back(2, W);
        }
    }
}
