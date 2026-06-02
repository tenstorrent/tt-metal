// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for per_token_cast_to_fp8 (step 4: real scale). Writes the e4m3 output column-block-major,
// and builds the fp32 scale output [H, W/128] by extracting column 0 of the compute's per-chunk
// scale tiles (cb_scale_tiles: GROUPS_PER_BLOCK tiles per col-block, tile g col 0 = scale[:, c*G+g]).
//
// The extracted per-row scales are accumulated (across all col-blocks of a tile-row) into a
// page-strided scratch, then written as full scale rows (page-aligned source -> aligned DRAM page)
// to satisfy NOC alignment (scale pages are round_up(W/128*4, 64); see notes §6).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t e4m3_addr = get_arg_val<uint32_t>(0);
    uint32_t scale_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(2);
    uint32_t num_col_blocks = get_arg_val<uint32_t>(3);
    uint32_t start_tile_row = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_e4m3 = get_compile_time_arg_val(0);
    constexpr uint32_t e4m3_col_block_bytes = get_compile_time_arg_val(1);  // 1024
    constexpr uint32_t cb_scale_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scale_scratch = get_compile_time_arg_val(3);
    constexpr uint32_t scale_groups = get_compile_time_arg_val(4);              // W / 128
    constexpr uint32_t scale_aligned_page_bytes = get_compile_time_arg_val(5);  // per-row scratch stride
    // Tile / face dims from the tensor's tile spec (32x32 / 16x16 by default; tiny tiles supported).
    constexpr uint32_t tile_h = get_compile_time_arg_val(6);
    constexpr uint32_t tile_w = get_compile_time_arg_val(7);
    constexpr uint32_t face_h = get_compile_time_arg_val(8);
    constexpr uint32_t face_w = get_compile_time_arg_val(9);
    constexpr uint32_t COL_BLOCK_ELEMS = 1024;
    constexpr uint32_t SCALE_GROUP_SIZE = 128;
    constexpr uint32_t GROUPS_PER_BLOCK = COL_BLOCK_ELEMS / SCALE_GROUP_SIZE;  // 8
    constexpr uint32_t face_elems = face_h * face_w;                           // fp32 per face
    constexpr uint32_t faces_per_row = tile_w / face_w;                        // face columns per tile
    constexpr uint32_t TILE_FP32 = tile_h * tile_w;                            // fp32 per tile
    constexpr uint32_t scale_row_bytes = scale_groups * 4;
    constexpr uint32_t scale_aligned_u32 = scale_aligned_page_bytes / 4;

    constexpr auto e4m3_args = TensorAccessorArgs<10>();
    constexpr auto scale_args = TensorAccessorArgs<e4m3_args.next_compile_time_args_offset()>();
    const auto e4m3 = TensorAccessor(e4m3_args, e4m3_addr);
    const auto scale = TensorAccessor(scale_args, scale_addr);

    uint32_t scratch = get_write_ptr(cb_scale_scratch);
    volatile tt_l1_ptr uint32_t* scratch_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        for (uint32_t c = 0; c < num_col_blocks; ++c) {
            // e4m3 output
            cb_wait_front(cb_e4m3, tile_h);
            uint32_t l1 = get_read_ptr(cb_e4m3);
            uint32_t col_offset_bytes = c * e4m3_col_block_bytes;
            for (uint32_t s = 0; s < tile_h; ++s) {
                uint32_t page_id = (start_tile_row + tr) * tile_h + s;
                noc_async_write(l1, e4m3.get_noc_addr(page_id) + col_offset_bytes, e4m3_col_block_bytes);
                l1 += e4m3_col_block_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_e4m3, tile_h);

            // extract column 0 of the GROUPS_PER_BLOCK scale tiles into the per-row scratch
            cb_wait_front(cb_scale_tiles, GROUPS_PER_BLOCK);
            volatile tt_l1_ptr uint32_t* tiles =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_scale_tiles));
            for (uint32_t g = 0; g < GROUPS_PER_BLOCK; ++g) {
                uint32_t tile_base = g * TILE_FP32;
                uint32_t global_group = c * GROUPS_PER_BLOCK + g;
                for (uint32_t s = 0; s < tile_h; ++s) {
                    // tile column 0 of row s: face (s/face_h, 0) at in-face row (s % face_h), col 0.
                    uint32_t col0_idx = (s / face_h) * faces_per_row * face_elems + (s % face_h) * face_w;
                    scratch_u32[s * scale_aligned_u32 + global_group] = tiles[tile_base + col0_idx];
                }
            }
            cb_pop_front(cb_scale_tiles, GROUPS_PER_BLOCK);
        }

        // write this tile-row's scale rows (full row, aligned)
        for (uint32_t s = 0; s < tile_h; ++s) {
            uint32_t page_id = (start_tile_row + tr) * tile_h + s;
            noc_async_write(scratch + s * scale_aligned_page_bytes, scale.get_noc_addr(page_id), scale_row_bytes);
        }
        noc_async_write_barrier();
    }
}
