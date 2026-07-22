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
// Compile-time args:
//   [0] cb_out_rm
//   [1] num_t_tiles, [2] T, [3] W, [4] W2
//   [5] y_tiles (= ceil(2W/32)), [6] uv_tiles (= ceil(W2/32))
//   [7..] TensorAccessorArgs for Y, U, V buffers
//
// Runtime args:
//   [0] y_addr, [1] u_addr, [2] v_addr, [3] unit_start, [4] unit_count

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
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
    constexpr auto y_args = TensorAccessorArgs<7>();
    constexpr auto u_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();

    constexpr uint32_t last_tile_elems = T - (num_t_tiles - 1) * TILE_W;
    constexpr uint32_t y_sticks = 2 * W;

    const auto sy = TensorAccessor(y_args, y_addr);
    const auto su = TensorAccessor(u_args, u_addr);
    const auto sv = TensorAccessor(v_args, v_addr);
    const Noc noc;

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

            cb_wait_front(cb_out_rm, 1);
            const uint32_t page_l1 = get_read_ptr(cb_out_rm);
            for (uint32_t s = 0; s < sticks; s++) {
                uint32_t spatial = base_spatial + base + s;
                uint32_t stick_l1 = page_l1 + s * TILE_W;
                noc.async_write(
                    CoreLocalMem<uint8_t>(stick_l1),
                    dst,
                    n_elems,
                    {},
                    {.page_id = spatial, .offset_bytes = byte_off_out});
                noc.async_writes_flushed();
            }
            cb_pop_front(cb_out_rm, 1);
        }
    };

    for (uint32_t u = unit_start; u < unit_start + unit_count; u++) {
        const uint32_t g = u / num_t_tiles;
        const uint32_t tt = u % num_t_tiles;
        const bool is_last_t = (tt == num_t_tiles - 1) && (last_tile_elems < TILE_W);
        const uint32_t n_elems = is_last_t ? last_tile_elems : TILE_W;
        const uint32_t byte_off_out = tt * TILE_W;

        write_plane(sy, 2 * g * W, y_sticks, y_tiles, byte_off_out, n_elems);  // Y: 2 rows
        write_plane(su, g * W2, W2, uv_tiles, byte_off_out, n_elems);          // Cb
        write_plane(sv, g * W2, W2, uv_tiles, byte_off_out, n_elems);          // Cr
    }

    noc.async_write_barrier();
}
