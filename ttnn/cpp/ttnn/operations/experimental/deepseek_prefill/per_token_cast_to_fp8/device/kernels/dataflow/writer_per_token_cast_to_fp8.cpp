// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for per_token_cast_to_fp8 (group-major). Mirror of the reader: the compute produces, per
// block, tile_h e4m3 group-rows (one per stream group) plus one fp32 scale tile whose column 0
// holds the tile_h per-group scales. The writer walks the same group stream and:
//   - writes each bank-contiguous run of e4m3 groups back to its row at column gir*128 (one
//     noc_async_write per run, mirroring the reader's contiguous reads);
//   - accumulates per-group scales into a persistent per-token scale-row scratch and flushes the
//     full row (page-aligned source -> aligned DRAM page) when the token's last group is emitted.
//     A block may straddle tokens and a token may straddle blocks, so the scratch and the
//     (current_row, current_col) cursor persist across blocks.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Extract column 0 of one fp32 tile into out[0..tile_h) (one value per group-row); face-aware walk.
template <uint32_t face_h, uint32_t face_w, uint32_t FACE_ROWS, uint32_t FACE_ROW_STRIDE>
static inline void extract_first_column(volatile tt_l1_ptr uint32_t* tile, uint32_t* out) {
    uint32_t s = 0;
    uint32_t face_base = 0;  // = face_row * FACE_ROW_STRIDE
    for (uint32_t fr = 0; fr < FACE_ROWS; ++fr) {
        uint32_t col0_idx = face_base;  // in-face row 0, col 0
        for (uint32_t r = 0; r < face_h; ++r) {
            out[s] = tile[col0_idx];
            col0_idx += face_w;
            ++s;
        }
        face_base += FACE_ROW_STRIDE;
    }
}

void kernel_main() {
    uint32_t e4m3_addr = get_arg_val<uint32_t>(0);
    uint32_t scale_addr = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);
    uint32_t start_row = get_arg_val<uint32_t>(3);  // absolute first row of this core's stream
    uint32_t num_rows = get_arg_val<uint32_t>(4);   // rows owned by this core
    uint32_t width = get_arg_val<uint32_t>(5);      // H (elements per row)

    constexpr uint32_t cb_e4m3 = get_compile_time_arg_val(0);
    constexpr uint32_t e4m3_group_bytes = get_compile_time_arg_val(1);  // 128 (1 byte/elem)
    constexpr uint32_t cb_scale_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scale_scratch = get_compile_time_arg_val(3);
    constexpr uint32_t scale_groups = get_compile_time_arg_val(4);              // H / 128 (full row width)
    constexpr uint32_t scale_aligned_page_bytes = get_compile_time_arg_val(5);  // per-row scratch footprint
    // Tile / face dims from the tensor's tile spec.
    constexpr uint32_t tile_h = get_compile_time_arg_val(6);
    constexpr uint32_t tile_w = get_compile_time_arg_val(7);
    constexpr uint32_t face_h = get_compile_time_arg_val(8);
    constexpr uint32_t face_w = get_compile_time_arg_val(9);
    constexpr uint32_t COL_BLOCK_ELEMS = 128;                         // column-block width = one group
    constexpr uint32_t COL_BLOCK_TILES = COL_BLOCK_ELEMS / tile_w;    // 4 tiles per block
    constexpr uint32_t face_elems = face_h * face_w;                  // fp32 per face
    constexpr uint32_t faces_per_row = tile_w / face_w;               // face columns per tile
    constexpr uint32_t FACE_ROWS = tile_h / face_h;                   // face rows per tile
    constexpr uint32_t FACE_ROW_STRIDE = faces_per_row * face_elems;  // fp32 stride per face row
    constexpr uint32_t scale_row_bytes = scale_groups * 4;

    constexpr auto e4m3_args = TensorAccessorArgs<10>();
    constexpr auto scale_args = TensorAccessorArgs<e4m3_args.next_compile_time_args_offset()>();
    const auto e4m3 = TensorAccessor(e4m3_args, e4m3_addr);
    const auto scale = TensorAccessor(scale_args, scale_addr);

    // Persistent per-token scale-row scratch (page-aligned), reused across blocks.
    uint32_t tok_scratch = get_write_ptr(cb_scale_scratch);
    volatile tt_l1_ptr uint32_t* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tok_scratch);

    uint32_t block_scales[tile_h];  // tile_h is compile-time constant

    const uint32_t groups_per_row = width / COL_BLOCK_ELEMS;  // H / 128
    const uint32_t total_groups = num_rows * groups_per_row;
    const uint32_t end_row = start_row + num_rows;
    uint32_t current_row = start_row;
    uint32_t current_col = 0;  // element offset within the current row

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t base = blk * tile_h;
        const uint32_t remaining = total_groups - base;
        const uint32_t real_in_block = remaining < tile_h ? remaining : tile_h;

        cb_wait_front(cb_e4m3, COL_BLOCK_TILES);
        uint32_t e4m3_l1 = get_read_ptr(cb_e4m3);
        cb_wait_front(cb_scale_tiles, 1);
        extract_first_column<face_h, face_w, FACE_ROWS, FACE_ROW_STRIDE>(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_scale_tiles)), block_scales);

        uint32_t br = 0;  // block-local group-row
        while (br < real_in_block && current_row < end_row) {
            uint32_t gir = current_col / COL_BLOCK_ELEMS;  // group index within the row
            uint32_t groups_left_in_row = groups_per_row - gir;
            uint32_t groups_left_in_block = real_in_block - br;
            uint32_t run = groups_left_in_row < groups_left_in_block ? groups_left_in_row : groups_left_in_block;

            // e4m3: write `run` contiguous groups (run*128 bytes) back to (current_row, current_col).
            noc_async_write(
                e4m3_l1 + br * e4m3_group_bytes, e4m3.get_noc_addr(current_row) + current_col, run * e4m3_group_bytes);
            // scale: stage this run's per-group scales into the token's scratch row.
            for (uint32_t g = 0; g < run; ++g) {
                tok[gir + g] = block_scales[br + g];
            }
            br += run;
            current_col += run * COL_BLOCK_ELEMS;
            if (current_col >= width) {  // token complete -> flush its full scale row (aligned)
                noc_async_write(tok_scratch, scale.get_noc_addr(current_row), scale_row_bytes);
                noc_async_write_barrier();  // scratch is reused by the next token
                current_col = 0;
                ++current_row;
            }
        }
        noc_async_write_barrier();  // drain this block's e4m3 writes before the CB page is reused
        cb_pop_front(cb_e4m3, COL_BLOCK_TILES);
        cb_pop_front(cb_scale_tiles, 1);
    }
}
