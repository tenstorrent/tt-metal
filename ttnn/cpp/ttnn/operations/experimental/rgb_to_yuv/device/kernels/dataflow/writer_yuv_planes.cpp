// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writer for tile-based YUV conversion (per-unit).
//
// Consumes uint8 row-major tile pages (32 sticks x 32 columns) from the compute
// kernel via cb_out_rm and NOC-writes them to the Y, U, V DRAM planes.  For
// each (row-group, T-tile) unit the compute kernel emits Y (2 rows) then Cb
// (1 UV row) then Cr, so the writer drains and writes them in that order.
//
// Two output layouts. Plain: planes are (1, H, W, T), one T-byte page per stick. Wide (`wide` = 1, one T tile
// per unit): planes are (1, H, W*T); a unit's rows are staged in L1 and written whole, W*T bytes per page.
//
// Compile-time args:
//   [0] cb_out_rm
//   [1] num_t_tiles, [2] T, [3] W, [4] W2
//   [5] y_tiles (= ceil(2W/32)), [6] uv_tiles (= ceil(W2/32))
//   [7] wide (0/1), [8] cb_rowbuf, [9] row_bytes_y (= W*T), [10] row_bytes_uv (= W2*T)
//   [11..] TensorAccessorArgs for Y, U, V buffers
//
// Runtime args:
//   [0] y_addr, [1] u_addr, [2] v_addr, [3] unit_start, [4] unit_count

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"

constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t u_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t unit_start = get_arg_val<uint32_t>(3);
    const uint32_t unit_count = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(0);
    constexpr uint32_t num_t_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t T = get_compile_time_arg_val(2);
    constexpr uint32_t W = get_compile_time_arg_val(3);
    constexpr uint32_t W2 = get_compile_time_arg_val(4);
    constexpr uint32_t y_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t uv_tiles = get_compile_time_arg_val(6);
    constexpr bool wide = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t cb_rowbuf = get_compile_time_arg_val(8);
    constexpr uint32_t row_bytes_y = get_compile_time_arg_val(9);
    constexpr uint32_t row_bytes_uv = get_compile_time_arg_val(10);
    constexpr auto y_args = TensorAccessorArgs<11>();
    constexpr auto u_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();

    constexpr uint32_t last_tile_elems = T - (num_t_tiles - 1) * TILE_W;
    constexpr uint32_t y_sticks = 2 * W;
    // One row buffer page per staged row, 64 B aligned so the row writes start aligned.
    constexpr uint32_t rowpage = ((row_bytes_y + 63) / 64) * 64;

    const auto sy = TensorAccessor(y_args, y_addr);
    const auto su = TensorAccessor(u_args, u_addr);
    const auto sv = TensorAccessor(v_args, v_addr);
    const Noc noc;
    CircularBuffer cb_out(cb_out_rm);
    CircularBuffer rowbuf(cb_rowbuf);
    const uint32_t rowbuf_base = wide ? rowbuf.get_write_ptr() : 0;

    // Drain `ntiles` output pages, writing `sticks_total` sticks starting at
    // output page `base_spatial`, at T-column offset `byte_off_out` (n_elems wide).
    auto write_plane = [&](const auto& dst,
                           uint32_t base_spatial,
                           uint32_t sticks_total,
                           uint32_t ntiles,
                           uint32_t byte_off_out,
                           uint32_t n_elems) {
        for (uint32_t tile = 0; tile < ntiles; tile++) {
            uint32_t base = tile * TILE_H;
            uint32_t sticks = (base + TILE_H <= sticks_total) ? TILE_H : (sticks_total - base);

            cb_out.wait_front(1);
            for (uint32_t s = 0; s < sticks; s++) {
                uint32_t spatial = base_spatial + base + s;
                // Source is cb_out's read pointer at this stick's byte offset.
                noc.async_write(
                    cb_out,
                    dst,
                    n_elems,
                    {.offset_bytes = s * TILE_W},
                    {.page_id = spatial, .offset_bytes = byte_off_out});
                noc.async_writes_flushed();
            }
            cb_out.pop_front(1);
        }
    };

    // Wide rows: stage `rows` complete rows of `sticks_per_row` sticks (T bytes each, one T tile) in
    // L1, then one write per row. Stick gs of the plane is row gs / sticks_per_row, column gs mod it.
    auto write_plane_wide = [&](const auto& dst,
                                uint32_t first_row,
                                uint32_t rows,
                                uint32_t sticks_per_row,
                                uint32_t ntiles,
                                uint32_t row_bytes) {
        const uint32_t sticks_total = rows * sticks_per_row;
        for (uint32_t tile = 0; tile < ntiles; tile++) {
            uint32_t base = tile * TILE_H;
            uint32_t sticks = (base + TILE_H <= sticks_total) ? TILE_H : (sticks_total - base);
            cb_out.wait_front(1);
            const uint32_t src_base = cb_out.get_read_ptr();
            for (uint32_t s = 0; s < sticks; s++) {
                const uint32_t gs = base + s;
                const uint32_t row = gs / sticks_per_row;
                const uint32_t col = gs - row * sticks_per_row;
                const uint32_t src = src_base + s * TILE_W;
                const uint32_t dst_l1 = rowbuf_base + row * rowpage + col * T;
                if constexpr (T % 4 == 0) {
                    volatile tt_l1_ptr uint32_t* s32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
                    volatile tt_l1_ptr uint32_t* d32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1);
                    for (uint32_t b = 0; b < T / 4; b++) {
                        d32[b] = s32[b];
                    }
                } else {
                    volatile tt_l1_ptr uint8_t* s8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src);
                    volatile tt_l1_ptr uint8_t* d8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst_l1);
                    for (uint32_t b = 0; b < T; b++) {
                        d8[b] = s8[b];
                    }
                }
            }
            cb_out.pop_front(1);
        }
        for (uint32_t row = 0; row < rows; row++) {
            noc.async_write(
                rowbuf, dst, row_bytes, {.offset_bytes = row * rowpage}, {.page_id = first_row + row, .offset_bytes = 0});
        }
        // The row buffer is reused by the next plane: wait until these writes have left L1.
        noc.async_writes_flushed();
    };

    for (uint32_t u = unit_start; u < unit_start + unit_count; u++) {
        const uint32_t g = u / num_t_tiles;
        const uint32_t tt = u % num_t_tiles;
        const bool is_last_t = (tt == num_t_tiles - 1) && (last_tile_elems < TILE_W);
        const uint32_t n_elems = is_last_t ? last_tile_elems : TILE_W;
        const uint32_t byte_off_out = tt * TILE_W;

        if constexpr (wide) {
            write_plane_wide(sy, 2 * g, 2, W, y_tiles, row_bytes_y);   // Y: rows 2g, 2g+1
            write_plane_wide(su, g, 1, W2, uv_tiles, row_bytes_uv);   // Cb: row g
            write_plane_wide(sv, g, 1, W2, uv_tiles, row_bytes_uv);   // Cr: row g
        } else {
            write_plane(sy, 2 * g * W, y_sticks, y_tiles, byte_off_out, n_elems);  // Y: 2 rows
            write_plane(su, g * W2, W2, uv_tiles, byte_off_out, n_elems);          // Cb
            write_plane(sv, g * W2, W2, uv_tiles, byte_off_out, n_elems);          // Cr
        }
    }

    noc.async_write_barrier();
}
