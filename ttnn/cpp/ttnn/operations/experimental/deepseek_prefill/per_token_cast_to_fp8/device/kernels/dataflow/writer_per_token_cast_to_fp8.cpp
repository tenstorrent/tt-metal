// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for per_token_cast_to_fp8. Mirror of the reader: the compute produces, per block, tile_h
// output_e4m3 block-rows plus one fp32 scale tile whose column 0
// holds the tile_h per-block scales. The writer walks the same block stream and:
//   - writes each bank-contiguous run of output_e4m3 blocks back to its row at the current block
//     column offset (one NoC async write per run, mirroring the reader's contiguous reads);
//   - accumulates per-block scales into a persistent per-token scale-row scratch and flushes the
//     full row (page-aligned source -> aligned DRAM page) when the token's last block is emitted.
//     A block may straddle tokens and a token may straddle blocks, so the scratch and the
//     (current_row, current_col) cursor persist across blocks.

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
    uint32_t output_e4m3_addr = get_arg_val<uint32_t>(0);
    uint32_t scale_addr = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);
    uint32_t start_row = get_arg_val<uint32_t>(3);  // absolute first row of this core's stream
    uint32_t num_rows = get_arg_val<uint32_t>(4);   // rows owned by this core
    uint32_t width = get_arg_val<uint32_t>(5);      // H (elements per row)

    constexpr uint32_t cb_output_e4m3 = get_compile_time_arg_val(0);
    constexpr uint32_t output_e4m3_block_bytes = get_compile_time_arg_val(1);  // 128 (1 byte/elem)
    constexpr uint32_t cb_scale_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scale_scratch = get_compile_time_arg_val(3);
    constexpr uint32_t scale_blocks_per_row = get_compile_time_arg_val(4);      // H / 128 (full row width)
    constexpr uint32_t scale_aligned_page_bytes = get_compile_time_arg_val(5);  // per-row scratch footprint
    // Tile / face dims from the tensor's tile spec.
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
    constexpr uint32_t scale_row_bytes = scale_blocks_per_row * 4;

    constexpr auto output_e4m3_accessor_args = TensorAccessorArgs<10>();
    constexpr auto scale_args = TensorAccessorArgs<output_e4m3_accessor_args.next_compile_time_args_offset()>();
    const auto output_e4m3 = TensorAccessor(output_e4m3_accessor_args, output_e4m3_addr);
    const auto scale = TensorAccessor(scale_args, scale_addr);
    Noc noc;
    CircularBuffer cb_output_e4m3_obj(cb_output_e4m3);
    CircularBuffer cb_scale_tiles_obj(cb_scale_tiles);
    CircularBuffer cb_scale_scratch_obj(cb_scale_scratch);

    // Persistent per-token scale-row scratch (page-aligned), reused across blocks.
    CoreLocalMem<volatile uint32_t> tok(cb_scale_scratch_obj.get_write_ptr());

    uint32_t block_scales[tile_h];  // tile_h is compile-time constant

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
}
