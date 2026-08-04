// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writer for tile-based YUV conversion (multicore).
//
// Receives uint8 row-major tile pages (32 sticks x 32 columns) from the
// compute kernel via cb_out_rm and NOC-writes them directly to DRAM.
// The bf16→uint8 typecast is done in the compute kernel via SFPU.
//
// Compile-time args:
//   [0] cb_out_rm     — uint8 row-major output from compute
//   [1] num_t_tiles
//   [2] T
//   [3..] TensorAccessorArgs for Y, U, V buffers
//
// Runtime args:
//   [0] y_addr, [1] u_addr, [2] v_addr
//   [3] y_start, [4] y_count
//   [5] uv_start, [6] uv_count

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t u_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t y_start = get_arg_val<uint32_t>(3);
    const uint32_t y_count = get_arg_val<uint32_t>(4);
    const uint32_t uv_start = get_arg_val<uint32_t>(5);
    const uint32_t uv_count = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(0);
    constexpr uint32_t num_t_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t T = get_compile_time_arg_val(2);
    constexpr auto y_args = TensorAccessorArgs<3>();
    constexpr auto u_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();

    constexpr uint32_t last_tile_elems = T - (num_t_tiles - 1) * TILE_W;

    const auto sy = TensorAccessor(y_args, y_addr);
    const auto su = TensorAccessor(u_args, u_addr);
    const auto sv = TensorAccessor(v_args, v_addr);

    auto write_plane = [&](const auto& dst, uint32_t start, uint32_t count) {
        uint32_t batches = (count + TILE_H - 1) / TILE_H;

        for (uint32_t batch = 0; batch < batches; batch++) {
            uint32_t base = start + batch * TILE_H;
            uint32_t sticks = (base + TILE_H <= start + count) ? TILE_H : (start + count - base);

            for (uint32_t tt = 0; tt < num_t_tiles; tt++) {
                const bool is_last = (tt == num_t_tiles - 1) && (last_tile_elems < TILE_W);
                const uint32_t n_elems = is_last ? last_tile_elems : TILE_W;
                const uint32_t byte_off_out = tt * TILE_W;

                cb_wait_front(cb_out_rm, 1);
                const uint32_t page_l1 = get_read_ptr(cb_out_rm);

                for (uint32_t s = 0; s < sticks; s++) {
                    uint32_t spatial = base + s;
                    uint32_t stick_l1 = page_l1 + s * TILE_W;
                    noc_async_write(stick_l1, dst.get_noc_addr(spatial, byte_off_out), n_elems);
                    noc_async_writes_flushed();
                }

                cb_pop_front(cb_out_rm, 1);
            }
        }
    };

    write_plane(sy, y_start, y_count);
    write_plane(su, uv_start, uv_count);
    write_plane(sv, uv_start, uv_count);

    noc_async_write_barrier();
}
