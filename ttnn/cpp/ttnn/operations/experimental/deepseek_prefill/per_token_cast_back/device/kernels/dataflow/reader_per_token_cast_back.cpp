// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for per_token_cast_back (group-major). Streams the core's rows as a flat sequence of
// 128-element groups; a block is tile_h consecutive groups (COL_BLOCK_TILES tiles after tilize).
// Per block:
//   - e4m3: read each bank-contiguous run of groups (within one row) into cb_e4m3 with a single
//     noc_async_read (mirrors the writer);
//   - scale: read the (few) tokens spanned by the block as full, page-aligned scale rows into a
//     reader-private scratch (whole-row reads keep the L1 destination congruent mod 64 with the
//     64-aligned DRAM page), then build the single bcast operand tile whose column 0 row r = the
//     scale of group-row r = scale[token_r][gir_r] (face-aware). The (unchanged, GROUPS_PER_BLOCK=1)
//     compute multiplies each group-row by its column-0 scale.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t e4m3_addr = get_arg_val<uint32_t>(0);
    uint32_t scale_addr = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);
    uint32_t start_row = get_arg_val<uint32_t>(3);  // absolute first row of this core's stream
    uint32_t num_rows = get_arg_val<uint32_t>(4);   // rows owned by this core
    uint32_t width = get_arg_val<uint32_t>(5);      // H (elements per row)

    constexpr uint32_t cb_e4m3 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scale_bcast = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scale_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t e4m3_group_bytes = get_compile_time_arg_val(3);          // 128 (1 byte/elem)
    constexpr uint32_t groups_per_block = get_compile_time_arg_val(4);          // 1
    constexpr uint32_t scale_aligned_page_bytes = get_compile_time_arg_val(5);  // aligned row footprint
    // Tile / face dims from the tensor's tile spec.
    constexpr uint32_t tile_h = get_compile_time_arg_val(6);
    constexpr uint32_t tile_w = get_compile_time_arg_val(7);
    constexpr uint32_t face_h = get_compile_time_arg_val(8);
    constexpr uint32_t face_w = get_compile_time_arg_val(9);
    constexpr uint32_t COL_BLOCK_ELEMS = 128;                       // column-block width = one group
    constexpr uint32_t COL_BLOCK_TILES = COL_BLOCK_ELEMS / tile_w;  // 4 tiles per block
    constexpr uint32_t face_elems = face_h * face_w;                // fp32 per face
    constexpr uint32_t faces_per_row = tile_w / face_w;             // face columns per tile
    constexpr uint32_t TILE_BYTES = tile_h * tile_w * sizeof(float);
    constexpr uint32_t FACE_ROWS = tile_h / face_h;                                         // face rows per tile
    constexpr uint32_t FACE_ROW_STRIDE_BYTES = faces_per_row * face_elems * sizeof(float);  // per face row
    constexpr uint32_t FACE_W_BYTES = face_w * sizeof(float);                               // per in-face row
    constexpr uint32_t scale_aligned_u32 = scale_aligned_page_bytes / 4;
    (void)groups_per_block;  // always 1 here (one bcast tile per block)

    constexpr auto e4m3_args = TensorAccessorArgs<10>();
    constexpr auto scale_args = TensorAccessorArgs<e4m3_args.next_compile_time_args_offset()>();
    const auto e4m3 = TensorAccessor(e4m3_args, e4m3_addr);
    const auto scale = TensorAccessor(scale_args, scale_addr);

    // Reader-private scratch: up to tile_h tokens' full scale rows (one slot per token in a block).
    cb_reserve_back(cb_scale_scratch, 1);
    uint32_t scratch = get_write_ptr(cb_scale_scratch);

    const uint32_t groups_per_row = width >> 7;  // H / 128 (COL_BLOCK_ELEMS = 128); one-time shift
    const uint32_t total_groups = num_rows * groups_per_row;
    const uint32_t end_row = start_row + num_rows;

    // Persistent (row, gir) cursor over the flat group stream: no per-group div/mod (expensive on
    // the Baby RISC-V); advance gir by the run and reset to the next row with a conditional.
    uint32_t current_row = start_row;
    uint32_t gir = 0;

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t base = blk * tile_h;
        const uint32_t remaining = total_groups - base;
        const uint32_t real_in_block = remaining < tile_h ? remaining : tile_h;
        const uint32_t block_start_row = current_row;
        const uint32_t block_start_gir = gir;

        // --- e4m3: fill the block as [tile_h group-rows x 128] via bank-contiguous runs ---
        cb_reserve_back(cb_e4m3, COL_BLOCK_TILES);
        uint32_t e4m3_l1 = get_write_ptr(cb_e4m3);
        uint32_t last_row = current_row;  // row of the last run = the block's last token
        {
            uint32_t row = current_row;
            uint32_t gg = gir;
            uint32_t slot = 0;
            while (slot < real_in_block && row < end_row) {
                uint32_t groups_left_in_row = groups_per_row - gg;
                uint32_t slots_left = real_in_block - slot;
                uint32_t run = groups_left_in_row < slots_left ? groups_left_in_row : slots_left;
                noc_async_read(
                    e4m3.get_noc_addr(row) + gg * e4m3_group_bytes,
                    e4m3_l1 + slot * e4m3_group_bytes,
                    run * e4m3_group_bytes);
                last_row = row;
                slot += run;
                gg += run;
                if (gg >= groups_per_row) {  // run never crosses a row boundary, so this is exact
                    gg = 0;
                    ++row;
                }
            }
            current_row = row;  // advance the persistent cursor for the next block
            gir = gg;
        }

        // --- scale: read the tokens spanned by this block as full page-aligned rows ---
        for (uint32_t t = block_start_row; t <= last_row; ++t) {
            noc_async_read(
                scale.get_noc_addr(t),
                scratch + (t - block_start_row) * scale_aligned_page_bytes,
                scale_aligned_page_bytes);
        }
        noc_async_read_barrier();

        // --- build the bcast operand: column 0 row r = scale[token_r][gir_r] (face-aware walk) ---
        // (tok_off, gir_b) cursor walks the block's group-rows in face order -> no per-row div/mod.
        cb_reserve_back(cb_scale_bcast, 1);
        uint32_t page = get_write_ptr(cb_scale_bcast);
        uint32_t tok_off = 0;  // (token - block_start_row) * scale_aligned_page_bytes
        uint32_t gir_b = block_start_gir;
        uint32_t s = 0;
        uint32_t face_base_off = 0;
        for (uint32_t fr = 0; fr < FACE_ROWS; ++fr) {
            uint32_t col0_off = face_base_off;
            for (uint32_t r = 0; r < face_h; ++r) {
                if (s < real_in_block) {
                    uint32_t val = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + tok_off + gir_b * 4);
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page + col0_off) = val;
                    ++gir_b;
                    if (gir_b >= groups_per_row) {
                        gir_b = 0;
                        tok_off += scale_aligned_page_bytes;
                    }
                }
                col0_off += FACE_W_BYTES;
                ++s;
            }
            face_base_off += FACE_ROW_STRIDE_BYTES;
        }
        cb_push_back(cb_scale_bcast, 1);
        cb_push_back(cb_e4m3, COL_BLOCK_TILES);
    }
}
