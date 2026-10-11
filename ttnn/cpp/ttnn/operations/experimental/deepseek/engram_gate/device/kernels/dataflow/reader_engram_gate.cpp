// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Stage this core's D-slice [c*w, (c+1)*w) of every (user, copy) row into padded row-major CBs:
//   x[r]     = streams row r of the slice's home shard  (the shard may be wider than w)
//   key[r]   = kv[b, h*D  + c*w : ... + w]               (r = b*hc + h)
//   value[r] = kv[b, hc*D + c*w : ... + w]               (the same row for every copy h)
// kv is width-sharded on its own grid, so a slice may straddle two kv shards.
void kernel_main() {
    constexpr uint32_t cb_x_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k_rm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v_rm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t rows = get_compile_time_arg_val(4);
    constexpr uint32_t rm_pages = get_compile_time_arg_val(5);
    constexpr uint32_t w = get_compile_time_arg_val(6);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t hc = get_compile_time_arg_val(8);
    constexpr uint32_t d = get_compile_time_arg_val(9);
    constexpr uint32_t kv_shard_w = get_compile_time_arg_val(10);
    constexpr uint32_t streams_row_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t row_bytes = w * elem_bytes;

    const uint32_t c = get_arg_val<uint32_t>(0);
    const uint32_t streams_addr = get_arg_val<uint32_t>(1);
    const uint32_t kv_addr = get_arg_val<uint32_t>(2);
    const uint32_t home_x = get_arg_val<uint32_t>(3);
    const uint32_t home_y = get_arg_val<uint32_t>(4);
    const uint32_t home_offset = get_arg_val<uint32_t>(5);

    // REDUCE scaler: 1.0 in the first row of each fp32 face.
    cb_reserve_back(cb_scaler, 1);
    volatile tt_l1_ptr uint32_t* scaler = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
    for (uint32_t i = 0; i < 1024; ++i) {
        scaler[i] = 0;
    }
    for (uint32_t face = 0; face < 4; ++face) {
        for (uint32_t j = 0; j < 16; ++j) {
            scaler[face * 256 + j] = 0x3F800000;  // 1.0f
        }
    }
    cb_push_back(cb_scaler, 1);

    cb_reserve_back(cb_x_rm, rm_pages);
    cb_reserve_back(cb_k_rm, rm_pages);
    cb_reserve_back(cb_v_rm, rm_pages);
    const uint32_t x_ptr = get_write_ptr(cb_x_rm);
    const uint32_t k_ptr = get_write_ptr(cb_k_rm);
    const uint32_t v_ptr = get_write_ptr(cb_v_rm);

    if constexpr (streams_row_bytes == row_bytes) {
        noc_async_read(get_noc_addr(home_x, home_y, streams_addr), x_ptr, rows * row_bytes);
    } else {
        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_read(
                get_noc_addr(home_x, home_y, streams_addr + r * streams_row_bytes + home_offset),
                x_ptr + r * row_bytes,
                row_bytes);
        }
    }

    auto read_kv_slice = [&](uint32_t b, uint32_t col, uint32_t dst) {
        uint32_t remaining = w;
        while (remaining > 0) {
            const uint32_t shard = col / kv_shard_w;
            const uint32_t offset = col - shard * kv_shard_w;
            const uint32_t len = (kv_shard_w - offset < remaining) ? (kv_shard_w - offset) : remaining;
            const uint32_t src = kv_addr + (b * kv_shard_w + offset) * elem_bytes;
            noc_async_read(
                get_noc_addr(get_common_arg_val<uint32_t>(2 * shard), get_common_arg_val<uint32_t>(2 * shard + 1), src),
                dst,
                len * elem_bytes);
            col += len;
            dst += len * elem_bytes;
            remaining -= len;
        }
    };

    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t b = r / hc;
        const uint32_t h = r - b * hc;
        read_kv_slice(b, h * d + c * w, k_ptr + r * row_bytes);
        read_kv_slice(b, hc * d + c * w, v_ptr + r * row_bytes);
    }
    noc_async_read_barrier();

    cb_push_back(cb_x_rm, rm_pages);
    cb_push_back(cb_k_rm, rm_pages);
    cb_push_back(cb_v_rm, rm_pages);
}
