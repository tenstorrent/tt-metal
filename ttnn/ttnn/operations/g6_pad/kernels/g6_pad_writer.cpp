// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_pad new_storage_tiles writer (RISCV_0 / NOC1).
// Block operations (op_design.md, Block schedule / new_storage_tiles):
//   N1 make_fill_page   -- fill_l1_range of fill_bits into cb_fill_tile, once per kernel
//   N3 write_fill_tiles -- fill-only output tiles of the block written from cb_fill_tile BEFORE the
//                          wait on cb_input_tiles (they do not depend on the input: stall shadow)
//   N4 patch_block      -- wait block_tiles; in each edge slot RISC-store fill_bits into rows
//                          [edge_row, 32) (last input tile row) and/or cols [edge_col, 32) (last input
//                          tile col), face-aware, in place (bit-exact, no unpack/pack)
//   N5 store_block      -- write each source-bearing slot to its output page; writes_flushed; pop
//   N6 drain_writes     -- noc_async_write_barrier (BB-03)
// Every output page is written exactly once.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "g6_pad_common.hpp"

namespace {

constexpr uint32_t FACE = 16;

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_fill_tile = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t out_tiles_h = get_compile_time_arg_val(5);
    constexpr uint32_t out_tiles_w = get_compile_time_arg_val(6);
    constexpr uint32_t out_lead1 = get_compile_time_arg_val(7);
    constexpr uint32_t in_tiles_h = get_compile_time_arg_val(8);
    constexpr uint32_t in_tiles_w = get_compile_time_arg_val(9);
    constexpr uint32_t in_lead0 = get_compile_time_arg_val(10);
    constexpr uint32_t in_lead1 = get_compile_time_arg_val(11);
    constexpr uint32_t edge_row = get_compile_time_arg_val(12);
    constexpr uint32_t edge_col = get_compile_time_arg_val(13);
    constexpr auto dst_args = TensorAccessorArgs<14>();

    constexpr uint32_t row_bytes = FACE * elem_bytes;
    constexpr uint32_t face_bytes = FACE * row_bytes;
    static_assert(tile_bytes == 4 * face_bytes, "tile geometry");

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t fill_bits = get_arg_val<uint32_t>(3);

    if (num_tiles == 0) {
        return;
    }

    const auto dst = TensorAccessor(dst_args, dst_addr, tile_bytes);
    using Walker =
        g6_pad::OutTileWalker<out_tiles_h, out_tiles_w, out_lead1, in_tiles_h, in_tiles_w, in_lead0, in_lead1>;

    // N1: fill page (scratch owned by this kernel only; never pushed).
    const uint32_t fill_base = get_write_ptr(cb_fill_tile);
    {
        DeviceZoneScopedN("make_fill_page");
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(fill_base, tile_bytes, fill_bits);
    }

    Walker walk(start_tile);
    uint32_t t = start_tile;
    const uint32_t num_blocks = (num_tiles + block_tiles - 1) / block_tiles;
    uint32_t remaining = num_tiles;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t extent = remaining < block_tiles ? remaining : block_tiles;

        // N3: fill-only tiles of this block, issued before the input wait.
        {
            DeviceZoneScopedN("write_fill_tiles");
            Walker w = walk;
            for (uint32_t k = 0; k < extent; ++k) {
                if (!w.source_bearing()) {
                    noc_async_write(fill_base, dst.get_noc_addr(t + k), tile_bytes);
                }
                w.advance();
            }
        }

        // N4: patch edge slots in place (zone includes the wait).
        {
            DeviceZoneScopedN("patch_block");
            cb_wait_front(cb_input_tiles, block_tiles);
            if constexpr (edge_row != 0 || edge_col != 0) {
                const uint32_t slot_base = get_read_ptr(cb_input_tiles);
                Walker w = walk;
                for (uint32_t k = 0; k < extent; ++k) {
                    if (w.source_bearing()) {
                        const uint32_t base = slot_base + k * tile_bytes;
                        if constexpr (edge_row != 0) {
                            if (w.last_input_tile_row()) {
                                if constexpr (edge_row < FACE) {
                                    // face 0 rows [edge_row,16), then face 1 rows [edge_row,16) + faces 2,3
                                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                        base + edge_row * row_bytes, (FACE - edge_row) * row_bytes, fill_bits);
                                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                        base + face_bytes + edge_row * row_bytes,
                                        (FACE - edge_row) * row_bytes + 2 * face_bytes,
                                        fill_bits);
                                } else {
                                    constexpr uint32_t r0 = edge_row - FACE;
                                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                        base + 2 * face_bytes + r0 * row_bytes, (FACE - r0) * row_bytes, fill_bits);
                                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                        base + 3 * face_bytes + r0 * row_bytes, (FACE - r0) * row_bytes, fill_bits);
                                }
                            }
                        }
                        if constexpr (edge_col != 0) {
                            if (w.last_input_tile_col()) {
                                if constexpr (edge_col < FACE) {
                                    // faces 1 and 3 whole; faces 0 and 2 columns [edge_col,16) of each row
                                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                        base + face_bytes, face_bytes, fill_bits);
                                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                        base + 3 * face_bytes, face_bytes, fill_bits);
                                    for (uint32_t r = 0; r < FACE; ++r) {
                                        dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                            base + r * row_bytes + edge_col * elem_bytes,
                                            (FACE - edge_col) * elem_bytes,
                                            fill_bits);
                                        dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                            base + 2 * face_bytes + r * row_bytes + edge_col * elem_bytes,
                                            (FACE - edge_col) * elem_bytes,
                                            fill_bits);
                                    }
                                } else {
                                    constexpr uint32_t c0 = edge_col - FACE;
                                    for (uint32_t r = 0; r < FACE; ++r) {
                                        dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                            base + face_bytes + r * row_bytes + c0 * elem_bytes,
                                            (FACE - c0) * elem_bytes,
                                            fill_bits);
                                        dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                                            base + 3 * face_bytes + r * row_bytes + c0 * elem_bytes,
                                            (FACE - c0) * elem_bytes,
                                            fill_bits);
                                    }
                                }
                            }
                        }
                    }
                    w.advance();
                }
            }
        }

        // N5: store source-bearing slots.
        {
            DeviceZoneScopedN("store_block");
            const uint32_t slot_base = get_read_ptr(cb_input_tiles);
            for (uint32_t k = 0; k < extent; ++k) {
                if (walk.source_bearing()) {
                    noc_async_write(slot_base + k * tile_bytes, dst.get_noc_addr(t + k), tile_bytes);
                }
                walk.advance();
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_input_tiles, block_tiles);
        }
        t += extent;
        remaining -= extent;
    }

    {
        DeviceZoneScopedN("drain_writes");
        noc_async_write_barrier();
    }
}
