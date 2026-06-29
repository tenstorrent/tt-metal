// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for per_token_cast_to_fp8. Outputs are ROW_MAJOR for both input layouts. Per block the compute
// produces tile_h e4m3 block-rows plus one fp32 scale tile whose column 0 holds the tile_h per-block
// scales (pulled out by extract_first_column).
//
// ROW_MAJOR input: walks the same flat block stream as the reader with a (current_row, current_col)
// cursor; e4m3 runs are written back per bank-contiguous span and each token's full scale row is flushed
// when its last block is emitted.
//
// INPUT_TILE_LAYOUT: work is split over 128-wide blocks; a core owns a contiguous global block range, so
// each block g maps to (row-tile g/spr, column-block g%spr) and the row-tiles it touches are consecutive,
// each owning a contiguous column-block range. e4m3 rows go to their folded RM output rows (tile-padding
// rows skipped); scales for a row-tile are staged and flushed as one contiguous run per row.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

// Extract column 0 of one fp32 tile into out[0..tile_h) (one value per block row); face-aware walk.
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
#ifdef INPUT_TILE_LAYOUT
    uint32_t output_e4m3_addr = get_arg_val<uint32_t>(0);
    uint32_t scale_addr = get_arg_val<uint32_t>(1);
    uint32_t block_offset = get_arg_val<uint32_t>(2);         // first global block of this core
    uint32_t num_blocks = get_arg_val<uint32_t>(3);           // blocks owned by this core
    uint32_t rows_per_batch = get_arg_val<uint32_t>(4);       // R = logical rows per batch (dim -2)
    uint32_t row_tiles_per_batch = get_arg_val<uint32_t>(5);  // ceil(R / tile_h)
#else
    uint32_t output_e4m3_addr = get_arg_val<uint32_t>(0);
    uint32_t scale_addr = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);
    uint32_t start_row = get_arg_val<uint32_t>(3);  // absolute first row of this core's stream
    uint32_t num_rows = get_arg_val<uint32_t>(4);   // rows owned by this core
    uint32_t width = get_arg_val<uint32_t>(5);      // H (elements per row)
#endif

    constexpr uint32_t cb_output_e4m3 = get_compile_time_arg_val(0);
    constexpr uint32_t output_e4m3_block_bytes = get_compile_time_arg_val(1);  // 128 (1 byte/elem)
    constexpr uint32_t cb_scale_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scale_scratch = get_compile_time_arg_val(3);
    constexpr uint32_t scale_blocks_per_row = get_compile_time_arg_val(4);      // spr = H / 128
    constexpr uint32_t scale_aligned_page_bytes = get_compile_time_arg_val(5);  // per-row scratch footprint
    constexpr uint32_t tile_h = get_compile_time_arg_val(6);
    constexpr uint32_t tile_w = get_compile_time_arg_val(7);
    constexpr uint32_t face_h = get_compile_time_arg_val(8);
    constexpr uint32_t face_w = get_compile_time_arg_val(9);
    constexpr uint32_t block_w = 128;
    constexpr uint32_t tiles_per_block = block_w / tile_w;
    constexpr uint32_t face_elems = face_h * face_w;                  // fp32 per face
    constexpr uint32_t faces_per_row = tile_w / face_w;               // face columns per tile
    constexpr uint32_t FACE_ROWS = tile_h / face_h;                   // face rows per tile
    constexpr uint32_t FACE_ROW_STRIDE = faces_per_row * face_elems;  // fp32 stride per face row

    constexpr auto output_e4m3_accessor_args = TensorAccessorArgs<10>();
    constexpr auto scale_args = TensorAccessorArgs<output_e4m3_accessor_args.next_compile_time_args_offset()>();
    const auto output_e4m3 = TensorAccessor(output_e4m3_accessor_args, output_e4m3_addr);
    const auto scale = TensorAccessor(scale_args, scale_addr);
    Noc noc;
    CircularBuffer cb_output_e4m3_obj(cb_output_e4m3);
    CircularBuffer cb_scale_tiles_obj(cb_scale_tiles);
    CircularBuffer cb_scale_scratch_obj(cb_scale_scratch);

    // Persistent scale scratch (writer-private), reused across blocks / row-tiles.
    CoreLocalMem<volatile uint32_t> tok(cb_scale_scratch_obj.get_write_ptr());
    uint32_t block_scales[tile_h];  // tile_h is compile-time constant

#ifdef INPUT_TILE_LAYOUT
    constexpr uint32_t scale_page_words = scale_aligned_page_bytes / 4;

    // Flush the accumulated scale rows for a row-tile over its owned column range [lo, hi].
    auto flush_scale = [&](uint32_t row_base, uint32_t real_rows, uint32_t lo, uint32_t hi) {
        const uint32_t run_bytes = (hi - lo + 1) * 4;
        for (uint32_t i = 0; i < real_rows; ++i) {
            noc.async_write(
                use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_scale_scratch_obj),
                scale,
                run_bytes,
                {.offset_bytes = i * scale_aligned_page_bytes + lo * 4},
                {.page_id = row_base + i, .offset_bytes = lo * 4});
        }
        noc.async_write_barrier();  // scratch is reused by the next row-tile
    };

    const uint32_t end_block = block_offset + num_blocks;
    uint32_t cur_rt = 0xFFFFFFFFu;  // sentinel: no row-tile open yet
    uint32_t cur_row_base = 0, cur_real_rows = 0, seg_lo = 0, seg_hi = 0;

    for (uint32_t g = block_offset; g < end_block; ++g) {
        const uint32_t rt = g / scale_blocks_per_row;
        const uint32_t cb = g % scale_blocks_per_row;

        if (rt != cur_rt) {
            if (cur_rt != 0xFFFFFFFFu) {
                flush_scale(cur_row_base, cur_real_rows, seg_lo, seg_hi);
            }
            // Map the row-tile to its batch and folded RM output rows (tile-padding rows skipped).
            const uint32_t batch = rt / row_tiles_per_batch;
            const uint32_t local_row_base = (rt % row_tiles_per_batch) * tile_h;
            cur_row_base = batch * rows_per_batch + local_row_base;
            const uint32_t rows_left = rows_per_batch - local_row_base;
            cur_real_rows = rows_left < tile_h ? rows_left : tile_h;
            cur_rt = rt;
            seg_lo = cb;
        }
        seg_hi = cb;

        cb_output_e4m3_obj.wait_front(tiles_per_block);
        cb_scale_tiles_obj.wait_front(1);
        extract_first_column<face_h, face_w, FACE_ROWS, FACE_ROW_STRIDE>(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_scale_tiles_obj.get_read_ptr()), block_scales);

        const uint32_t col_offset = cb * block_w;
        for (uint32_t i = 0; i < cur_real_rows; ++i) {
            noc.async_write(
                cb_output_e4m3_obj,
                output_e4m3,
                output_e4m3_block_bytes,
                {.offset_bytes = i * output_e4m3_block_bytes},
                {.page_id = cur_row_base + i, .offset_bytes = col_offset});
            tok[i * scale_page_words + cb] = block_scales[i];
        }
        noc.async_write_barrier();  // drain e4m3 writes before the CB page is reused
        cb_output_e4m3_obj.pop_front(tiles_per_block);
        cb_scale_tiles_obj.pop_front(1);
    }

    if (cur_rt != 0xFFFFFFFFu) {
        flush_scale(cur_row_base, cur_real_rows, seg_lo, seg_hi);
    }
#else
    constexpr uint32_t scale_row_bytes = scale_blocks_per_row * 4;

    const uint32_t blocks_per_row = width >> 7;  // H / 128 (block_w = 128)
    const uint32_t total_blocks = num_rows * blocks_per_row;
    const uint32_t end_row = start_row + num_rows;
    uint32_t current_row = start_row;
    uint32_t current_col = 0;  // element offset within the current row

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t base = blk * tile_h;
        const uint32_t remaining = total_blocks - base;
        const uint32_t real_in_block = remaining < tile_h ? remaining : tile_h;

        cb_output_e4m3_obj.wait_front(tiles_per_block);
        cb_scale_tiles_obj.wait_front(1);
        extract_first_column<face_h, face_w, FACE_ROWS, FACE_ROW_STRIDE>(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_scale_tiles_obj.get_read_ptr()), block_scales);

        uint32_t br = 0;  // block-local row
        while (br < real_in_block && current_row < end_row) {
            uint32_t block_idx_in_row = current_col >> 7;  // block index within the row (block_w = 128)
            uint32_t blocks_left_in_row = blocks_per_row - block_idx_in_row;
            uint32_t blocks_left_in_batch = real_in_block - br;
            uint32_t run = blocks_left_in_row < blocks_left_in_batch ? blocks_left_in_row : blocks_left_in_batch;

            // output_e4m3: write `run` contiguous 128-element blocks back to (current_row, current_col).
            noc.async_write(
                cb_output_e4m3_obj,
                output_e4m3,
                run * output_e4m3_block_bytes,
                {.offset_bytes = br * output_e4m3_block_bytes},
                {.page_id = current_row, .offset_bytes = current_col});
            // scale: stage this run's per-block scales into the token's scratch row.
            for (uint32_t g = 0; g < run; ++g) {
                tok[block_idx_in_row + g] = block_scales[br + g];
            }
            br += run;
            current_col += run * block_w;
            if (current_col >= width) {  // token complete -> flush its full scale row (aligned)
                noc.async_write(
                    use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_scale_scratch_obj),
                    scale,
                    scale_row_bytes,
                    {.offset_bytes = 0},
                    {.page_id = current_row});
                noc.async_write_barrier();  // scratch is reused by the next token
                current_col = 0;
                ++current_row;
            }
        }
        noc.async_write_barrier();  // drain this block's output_e4m3 writes before the CB page is reused
        cb_output_e4m3_obj.pop_front(tiles_per_block);
        cb_scale_tiles_obj.pop_front(1);
    }
#endif
}
