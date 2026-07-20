// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for per_token_cast_back. Streams the core's rows as a flat sequence of 128-element scale
// blocks; a block is tile_h consecutive scale blocks (tiles_per_block tiles after tilize).
// Per block:
//   - input_e4m3: read each bank-contiguous run of scale blocks into cb_input_e4m3 with a single
//     NoC async read (mirrors the writer);
//   - scale: read the (few) tokens spanned by the block as full, page-aligned scale rows into a
//     reader-private scratch (whole-row reads keep the L1 destination congruent mod 64 with the
//     64-aligned DRAM page), then build the single bcast operand tile whose column 0 row r = the
//     scale of block row r = scale[token_r][block_r] (face-aware). The compute multiplies each
//     block row by its column-0 scale.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t input_e4m3_addr = get_arg_val<uint32_t>(0);
    uint32_t scale_addr = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);
    uint32_t start_row = get_arg_val<uint32_t>(3);  // absolute first row of this core's stream
    uint32_t num_rows = get_arg_val<uint32_t>(4);   // rows owned by this core
    uint32_t width = get_arg_val<uint32_t>(5);      // H (elements per row)

    constexpr uint32_t cb_input_e4m3 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scale_bcast = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scale_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t input_e4m3_block_bytes = get_compile_time_arg_val(3);    // 128 (1 byte/elem)
    constexpr uint32_t block_ht = get_compile_time_arg_val(4);                  // BlockHt = 1
    constexpr uint32_t scale_aligned_page_bytes = get_compile_time_arg_val(5);  // aligned row footprint
    // Tile / face dims from the tensor's tile spec.
    constexpr uint32_t tile_h = get_compile_time_arg_val(6);
    constexpr uint32_t tile_w = get_compile_time_arg_val(7);
    constexpr uint32_t face_h = get_compile_time_arg_val(8);
    constexpr uint32_t face_w = get_compile_time_arg_val(9);
    // Scale is FLOAT32 (4 bytes/elem); the bcast operand keeps the fp32 scale (raw copy, no conversion).
    constexpr uint32_t scale_elem_shift = 2;  // byte offset -> fp32 word index
    constexpr uint32_t block_w = 128;
    constexpr uint32_t tiles_per_block = block_w / tile_w;
    constexpr uint32_t face_elems = face_h * face_w;                // scale elems per face
    constexpr uint32_t faces_per_row = tile_w / face_w;             // face columns per tile
    constexpr uint32_t FACE_ROWS = tile_h / face_h;                 // face rows per tile
    constexpr uint32_t FACE_ROW_STRIDE_BYTES = faces_per_row * face_elems * 4;  // fp32 bytes per face row
    constexpr uint32_t FACE_W_BYTES = face_w * 4;                               // fp32 bytes per in-face row

    (void)block_ht;  // kept as a compile-time layout arg for tensor accessor offset stability

    constexpr auto input_e4m3_accessor_args = TensorAccessorArgs<10>();
    constexpr auto scale_args = TensorAccessorArgs<input_e4m3_accessor_args.next_compile_time_args_offset()>();
    const auto input_e4m3 = TensorAccessor(input_e4m3_accessor_args, input_e4m3_addr);
    const auto scale = TensorAccessor(scale_args, scale_addr);
    Noc noc;
    CircularBuffer cb_input_e4m3_obj(cb_input_e4m3);
    CircularBuffer cb_scale_bcast_obj(cb_scale_bcast);
    CircularBuffer cb_scale_scratch_obj(cb_scale_scratch);

    // Reader-private scratch: up to tile_h tokens' full scale rows (one slot per token in a block).
    cb_scale_scratch_obj.reserve_back(1);
    const uint32_t scratch = cb_scale_scratch_obj.get_write_ptr();

    const uint32_t blocks_per_row = width >> 7;  // H / 128 (block_w = 128); one-time shift
    const uint32_t total_blocks = num_rows * blocks_per_row;
    const uint32_t end_row = start_row + num_rows;

    // Persistent (row, block_idx) cursor over the flat block stream: no per-block div/mod (expensive
    // on the Baby RISC-V); advance block_idx by the run and reset to the next row with a conditional.
    uint32_t current_row = start_row;
    uint32_t block_idx_in_row = 0;

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t base = blk * tile_h;
        const uint32_t remaining = total_blocks - base;
        const uint32_t real_in_block = remaining < tile_h ? remaining : tile_h;
        const uint32_t block_start_row = current_row;
        const uint32_t block_start_idx = block_idx_in_row;

        // --- input_e4m3: fill the block as [tile_h scale-block rows x 128] via bank-contiguous runs ---
        cb_input_e4m3_obj.reserve_back(tiles_per_block);
        uint32_t last_row = current_row;  // row of the last run = the block's last token
        {
            uint32_t row = current_row;
            uint32_t block_idx = block_idx_in_row;
            uint32_t slot = 0;
            while (slot < real_in_block && row < end_row) {
                uint32_t blocks_left_in_row = blocks_per_row - block_idx;
                uint32_t slots_left = real_in_block - slot;
                uint32_t run = blocks_left_in_row < slots_left ? blocks_left_in_row : slots_left;
                noc.async_read(
                    input_e4m3,
                    cb_input_e4m3_obj,
                    run * input_e4m3_block_bytes,
                    {.page_id = row, .offset_bytes = block_idx * input_e4m3_block_bytes},
                    {.offset_bytes = slot * input_e4m3_block_bytes});
                last_row = row;
                slot += run;
                block_idx += run;
                if (block_idx >= blocks_per_row) {  // run never crosses a row boundary, so this is exact
                    block_idx = 0;
                    ++row;
                }
            }
            current_row = row;  // advance the persistent cursor for the next block
            block_idx_in_row = block_idx;
        }

        // --- scale: read the tokens spanned by this block as full page-aligned rows ---
        for (uint32_t t = block_start_row; t <= last_row; ++t) {
            noc.async_read(
                scale,
                use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_scale_scratch_obj),
                scale_aligned_page_bytes,
                {.page_id = t},
                {.offset_bytes = (t - block_start_row) * scale_aligned_page_bytes});
        }
        noc.async_read_barrier();

        // --- build the bcast operand: column 0 row r = scale[token_r][block_r] (face-aware walk) ---
        // (tok_off, block_idx_b) cursor walks the block rows in face order -> no per-row div/mod. The
        // operand keeps the fp32 scale (raw copy, no conversion), so the multiply's SrcB matches the
        // fp32 input tile.
        cb_scale_bcast_obj.reserve_back(1);
        const uint32_t page_ptr = cb_scale_bcast_obj.get_write_ptr();
        auto build_bcast = [&](auto scratch_mem, auto page) {
            uint32_t tok_off = 0;  // (token - block_start_row) * scale_aligned_page_bytes
            uint32_t block_idx_b = block_start_idx;
            uint32_t s = 0;
            uint32_t face_base_off = 0;
            for (uint32_t fr = 0; fr < FACE_ROWS; ++fr) {
                uint32_t col0_off = face_base_off;
                for (uint32_t r = 0; r < face_h; ++r) {
                    if (s < real_in_block) {
                        page[col0_off >> scale_elem_shift] = scratch_mem[(tok_off >> scale_elem_shift) + block_idx_b];
                        ++block_idx_b;
                        if (block_idx_b >= blocks_per_row) {
                            block_idx_b = 0;
                            tok_off += scale_aligned_page_bytes;
                        }
                    }
                    col0_off += FACE_W_BYTES;
                    ++s;
                }
                face_base_off += FACE_ROW_STRIDE_BYTES;
            }
        };
        build_bcast(CoreLocalMem<volatile uint32_t>(scratch), CoreLocalMem<volatile uint32_t>(page_ptr));
        cb_scale_bcast_obj.push_back(1);
        cb_input_e4m3_obj.push_back(tiles_per_block);
    }
}
