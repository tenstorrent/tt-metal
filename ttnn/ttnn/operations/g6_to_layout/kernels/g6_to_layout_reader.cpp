// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_to_layout reader (NCRISC, NoC0) — `load_block` of regime R1 (+ R1a live-face reads).
//
// A unit is one tile row of one width chunk (block_tile_cols tiles); a block is block_tile_rows
// consecutive units of this core. Per block: one cb_reserve_back, every live tile's reads issued,
// ONE read barrier, one push of the whole block. Tiles absent from a ragged last chunk and faces
// not read for tail tiles (R1a) leave stale L1 in their nominal slot — untilized, never written.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tile_cols = get_compile_time_arg_val(1);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(3);
    constexpr uint32_t num_col_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t tiles_per_image_h = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr bool w_tail_half = get_compile_time_arg_val(7) != 0;  // last tile col: faces {0,2}
    constexpr bool h_tail_half = get_compile_time_arg_val(8) != 0;  // last tile row per image: faces {0,1}
    constexpr auto input_args = TensorAccessorArgs<9>();

    constexpr uint32_t face_bytes = tile_bytes / 4;

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t units_this_core = get_arg_val<uint32_t>(2);

    const auto input_accessor = TensorAccessor(input_args, input_addr, tile_bytes);

    uint32_t tile_row = start_unit / num_col_chunks;
    uint32_t chunk = start_unit % num_col_chunks;

    for (uint32_t done = 0; done < units_this_core; done += block_tile_rows) {
        DeviceZoneScopedN("load_block");
        const uint32_t extent = (units_this_core - done) < block_tile_rows ? (units_this_core - done) : block_tile_rows;
        const uint32_t block_pages = extent * block_tile_cols;
        cb_reserve_back(cb_input_tiles, block_pages);
        uint32_t l1_unit = get_write_ptr(cb_input_tiles);

        for (uint32_t u = 0; u < extent; ++u) {
            const uint32_t first_col = chunk * block_tile_cols;
            const uint32_t cols_left = tensor_tile_cols - first_col;
            const uint32_t live_tiles = cols_left < block_tile_cols ? cols_left : block_tile_cols;
            const bool h_tail = h_tail_half && ((tile_row % tiles_per_image_h) == tiles_per_image_h - 1);
            uint32_t tile_id = tile_row * tensor_tile_cols + first_col;
            uint32_t l1_tile = l1_unit;
            for (uint32_t j = 0; j < live_tiles; ++j, ++tile_id, l1_tile += tile_bytes) {
                const bool w_tail = w_tail_half && (first_col + j == tensor_tile_cols - 1);
                const uint64_t src = input_accessor.get_noc_addr(tile_id);
                if (w_tail && h_tail) {
                    noc_async_read(src, l1_tile, face_bytes);  // face 0
                } else if (w_tail) {
                    noc_async_read(src, l1_tile, face_bytes);                                    // face 0
                    noc_async_read(src + 2 * face_bytes, l1_tile + 2 * face_bytes, face_bytes);  // face 2
                } else if (h_tail) {
                    noc_async_read(src, l1_tile, 2 * face_bytes);  // faces 0 + 1
                } else {
                    noc_async_read(src, l1_tile, tile_bytes);
                }
            }
            l1_unit += block_tile_cols * tile_bytes;
            if (++chunk == num_col_chunks) {
                chunk = 0;
                ++tile_row;
            }
        }
        {
            DeviceZoneScopedN("read_barrier");
            noc_async_read_barrier();
        }
        cb_push_back(cb_input_tiles, block_pages);
    }
}
