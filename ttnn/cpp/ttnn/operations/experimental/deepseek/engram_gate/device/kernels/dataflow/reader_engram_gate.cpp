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
    // > 0: streams / kv are (block_h x 32) tiles; row s * block_h + j of the 32x32 tiles is stream s of row j.
    constexpr uint32_t block_h = get_compile_time_arg_val(13);
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

    if constexpr (block_h > 0) {
        // Each (block_h x 32) tile's two faces (block_h rows of 16 columns each) land as rows
        // [s * block_h, (s + 1) * block_h) of the matching faces of 32x32 tile t: 2 reads per (s, t).
        constexpr uint32_t face_rows_bytes = block_h * 16 * elem_bytes;
        constexpr uint32_t small_tile_bytes = 2 * face_rows_bytes;
        constexpr uint32_t big_tile_bytes = 32 * 32 * elem_bytes;
        constexpr uint32_t big_face_bytes = 16 * 16 * elem_bytes;
        constexpr uint32_t big_face_row_bytes = 16 * elem_bytes;
        constexpr uint32_t wt = w / 32;
        constexpr uint32_t streams_tiles = streams_row_bytes / (32 * elem_bytes);
        constexpr uint32_t kv_tiles = kv_shard_w / 32;
        const uint32_t home_tile = home_offset / (32 * elem_bytes);
        // Byte offset of rows [s * block_h, ...) of the left face within a 32x32 tile (the right face is
        // big_face_bytes further); block_h divides 16, so a stream's rows never straddle two face rows.
        auto place = [&](uint32_t s) {
            const uint32_t r0 = s * block_h;
            return (r0 < 16 ? 0 : 2 * big_face_bytes) + (r0 % 16) * big_face_row_bytes;
        };
        auto read_small = [&](uint64_t src, uint32_t dst_tile, uint32_t s) {
            const uint32_t dst = dst_tile + place(s);
            noc_async_read(src, dst, face_rows_bytes);
            noc_async_read(src + face_rows_bytes, dst + big_face_bytes, face_rows_bytes);
        };
        auto kv_tile_noc = [&](uint32_t col) {
            const uint32_t tile = col / 32;
            const uint32_t shard = tile / kv_tiles;
            return get_noc_addr(
                get_common_arg_val<uint32_t>(2 * shard),
                get_common_arg_val<uint32_t>(2 * shard + 1),
                kv_addr + (tile - shard * kv_tiles) * small_tile_bytes);
        };
        for (uint32_t t = 0; t < wt; ++t) {
            const uint32_t x_tile = x_ptr + t * big_tile_bytes;
            const uint32_t k_tile = k_ptr + t * big_tile_bytes;
            const uint32_t v_tile = v_ptr + t * big_tile_bytes;
            const uint64_t v_src = kv_tile_noc(hc * d + c * w + t * 32);
            for (uint32_t s = 0; s < hc; ++s) {
                read_small(
                    get_noc_addr(home_x, home_y, streams_addr + (s * streams_tiles + home_tile + t) * small_tile_bytes),
                    x_tile,
                    s);
                read_small(kv_tile_noc(s * d + c * w + t * 32), k_tile, s);
                read_small(v_src, v_tile, s);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_x_rm, rm_pages);
        cb_push_back(cb_k_rm, rm_pages);
        cb_push_back(cb_v_rm, rm_pages);
        return;
    }

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
