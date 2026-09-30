// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// mhc_pre_xing reader (NCRISC, NoC 0). Per segment (token tile-row r): the row tile r of the coefficient source
// (the all-reduced [mix 24 | sum x^2 | 0..] row, or the finished [pre | post | comb] row) -> cb_row; with streams,
// then per y column c of the segment the n stream tiles X_i[r, c] -> cb_x (one push of n tiles per column).
// pack_stats: per token tile-row r, the partial mix tile r -> cb_row, then all n*Ct stream tiles of row r in order,
// pushed in chunks of `chunk` tiles.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "mhc_pre_xing_common.hpp"

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);  // C / 32 (units per row with streams)
    constexpr bool has_streams = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t cb_row = get_compile_time_arg_val(3);
    constexpr uint32_t cb_x = get_compile_time_arg_val(4);
    constexpr uint32_t row_page = get_compile_time_arg_val(5);
    constexpr uint32_t x_page = get_compile_time_arg_val(6);
    constexpr bool pack_stats = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t chunk = get_compile_time_arg_val(8);
    constexpr auto row_args = TensorAccessorArgs<9>();
    constexpr auto x_args = TensorAccessorArgs<row_args.next_compile_time_args_offset()>();

    const uint32_t row_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);

    const auto row_acc = TensorAccessor(row_args, row_addr, row_page);
    const auto x_acc = TensorAccessor(x_args, x_addr, x_page);

    if constexpr (pack_stats) {
        constexpr uint32_t k_tiles = n * ct;
        for (uint32_t row = start; row < start + count; ++row) {
            cb_reserve_back(cb_row, 1);
            noc_async_read_page(row, row_acc, get_write_ptr(cb_row));
            noc_async_read_barrier();
            cb_push_back(cb_row, 1);
            for (uint32_t k0 = 0; k0 < k_tiles; k0 += chunk) {
                cb_reserve_back(cb_x, chunk);
                uint32_t dst = get_write_ptr(cb_x);
                for (uint32_t k = 0; k < chunk; ++k) {
                    noc_async_read_page(row * k_tiles + k0 + k, x_acc, dst);
                    dst += x_page;
                }
                noc_async_read_barrier();
                cb_push_back(cb_x, chunk);
            }
        }
        return;
    }
    mhc_xing::SegmentWalker walker(start, count, has_streams ? ct : 1);
    while (!walker.done()) {
        const mhc_xing::Segment seg = walker.next();
        cb_reserve_back(cb_row, 1);
        noc_async_read_page(seg.row, row_acc, get_write_ptr(cb_row));
        noc_async_read_barrier();
        cb_push_back(cb_row, 1);
        if constexpr (has_streams) {
            const uint32_t row_base = seg.row * n * ct;
            for (uint32_t c = seg.col0; c < seg.col0 + seg.cols; ++c) {
                cb_reserve_back(cb_x, n);
                uint32_t dst = get_write_ptr(cb_x);
                for (uint32_t i = 0; i < n; ++i) {
                    noc_async_read_page(row_base + i * ct + c, x_acc, dst);
                    dst += x_page;
                }
                noc_async_read_barrier();
                cb_push_back(cb_x, n);
            }
        }
    }
}
