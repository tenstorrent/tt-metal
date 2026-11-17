// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm writer (BRISC). TILE: writes tile pages (Rb x Sb window per barrier).
// RM: writes only valid sticks / valid bytes of each stick segment (32 sticks per barrier);
// H/W pad is never written.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_tile_row = get_arg_val<uint32_t>(1);
    const uint32_t rows_core = get_arg_val<uint32_t>(2);

    constexpr bool IS_RM = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t ES = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t Sb = get_compile_time_arg_val(3);
    constexpr uint32_t Rb = get_compile_time_arg_val(4);
    constexpr uint32_t W = get_compile_time_arg_val(5);
    constexpr uint32_t H = get_compile_time_arg_val(6);
    constexpr uint32_t Ht_img = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(9);
    constexpr auto o_args = TensorAccessorArgs<10>();

    constexpr uint32_t TILE_DIM = 32;
    constexpr uint32_t num_seg = Wt / Sb;
    constexpr uint32_t SEG_BYTES = Sb * TILE_DIM * ES;

    DataflowBuffer dfb_tiles(cb_out_tiles);
    DataflowBuffer dfb_rm(cb_out_rm);

    const uint32_t tile_bytes = dfb_tiles.get_tile_size();
    const auto acc = TensorAccessor(o_args, o_addr, IS_RM ? W * ES : tile_bytes);

    for (uint32_t b = 0; b < rows_core; b += Rb) {
        const uint32_t rb = (rows_core - b) < Rb ? (rows_core - b) : Rb;
        const uint32_t tr0 = first_tile_row + b;
        for (uint32_t seg = 0; seg < num_seg; ++seg) {
            if constexpr (IS_RM) {
                for (uint32_t rr = 0; rr < rb; ++rr) {
                    const uint32_t tr = tr0 + rr;
                    const uint32_t img = tr / Ht_img;
                    const uint32_t r = tr % Ht_img;
                    const uint32_t valid_rows = (H - r * TILE_DIM) < TILE_DIM ? (H - r * TILE_DIM) : TILE_DIM;
                    const uint32_t stick0 = img * H + r * TILE_DIM;
                    dfb_rm.wait_front(Sb);
                    const uint32_t base = dfb_rm.get_read_ptr();
                    const uint32_t off = seg * SEG_BYTES;
                    const uint32_t avail = W * ES - off;
                    const uint32_t bytes = avail < SEG_BYTES ? avail : SEG_BYTES;
                    for (uint32_t k = 0; k < valid_rows; ++k) {
                        noc_async_write(base + k * SEG_BYTES, acc.get_noc_addr(stick0 + k, off), bytes);
                    }
                    noc_async_write_barrier();
                    dfb_rm.pop_front(Sb);
                }
            } else {
                dfb_tiles.wait_front(rb * Sb);
                const uint32_t base = dfb_tiles.get_read_ptr();
                for (uint32_t rr = 0; rr < rb; ++rr) {
                    for (uint32_t c = 0; c < Sb; ++c) {
                        noc_async_write_tile((tr0 + rr) * Wt + seg * Sb + c, acc, base + (rr * Sb + c) * tile_bytes);
                    }
                }
                noc_async_write_barrier();
                dfb_tiles.pop_front(rb * Sb);
            }
        }
    }
}
