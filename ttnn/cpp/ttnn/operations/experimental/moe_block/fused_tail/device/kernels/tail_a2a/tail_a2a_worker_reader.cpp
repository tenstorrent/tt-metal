// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TailA2A worker reader (RISCV_1, NoC0): my column slice of the owners' rows (row-major, 32 rows a row tile) and se's
// tiles, row tile by row tile.

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_loop.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_tail/device/kernels/tail_a2a/tail_rows.hpp"

MB_LOOP_BODY {
    constexpr uint32_t W = get_compile_time_arg_val(0), SE = get_compile_time_arg_val(1);
    constexpr auto y_args = TensorAccessorArgs<2>();
    constexpr auto s_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    const uint32_t y_addr = get_arg_val<uint32_t>(0), s_addr = get_arg_val<uint32_t>(1), c0 = get_arg_val<uint32_t>(2);
    const uint32_t RT = get_arg_val<uint32_t>(3), NTH = get_arg_val<uint32_t>(4);
    const auto y = TensorAccessor(y_args, y_addr, NTH * 64);
    const auto s = TensorAccessor(s_args, s_addr, 2048);
#ifdef TL_SKIP
    // (my live row tiles: to the compute in a CB_LIVE page, words 0 / 1)
    cb_reserve_back(CB_LIVE, 1);
    const uint32_t lw = get_write_ptr(CB_LIVE);
    const uint64_t live = tl_live(5, lw, RT);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lw)[0] = static_cast<uint32_t>(live);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lw)[1] = static_cast<uint32_t>(live >> 32);
    cb_push_back(CB_LIVE, 1);
#else
    constexpr uint64_t live = 0;
#endif
    for (uint32_t r = 0; r < RT; ++r) {
        if (!tl_is_live(live, r)) {
            continue;
        }
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

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_loop_main.hpp"
