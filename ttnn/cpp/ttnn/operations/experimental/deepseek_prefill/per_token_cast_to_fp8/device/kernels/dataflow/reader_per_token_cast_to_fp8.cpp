// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for per_token_cast_to_fp8. Fills the reduce scaler tile once, then streams
// the core's rows as a flat sequence of 128-element scale blocks. A block is tile_h consecutive blocks
// (block_capacity = tile_h * 128 elements = tiles_per_block tiles). Each bank-contiguous run (a span
// within one row) is read with a single NoC async read, so we exploit row-major DRAM locality
// instead of hopping banks. After tilize, the compute reduces each 128-element block independently.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);
    uint32_t start_row = get_arg_val<uint32_t>(2);  // absolute first row of this core's stream
    uint32_t num_rows = get_arg_val<uint32_t>(3);   // rows owned by this core
    uint32_t width = get_arg_val<uint32_t>(4);      // H (elements per row)

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t input_block_bytes = get_compile_time_arg_val(1);  // 128 * elem_size
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    // Tile / face dims from the tensor's tile spec.
    constexpr uint32_t tile_h = get_compile_time_arg_val(3);
    constexpr uint32_t tile_w = get_compile_time_arg_val(4);
    constexpr uint32_t face_h = get_compile_time_arg_val(5);
    constexpr uint32_t face_w = get_compile_time_arg_val(6);
    constexpr uint32_t ONE_F32_BITS = 0x3f800000u;  // 1.0f
    constexpr uint32_t face_elems = face_h * face_w;                       // fp32 elements per face
    constexpr uint32_t num_faces = (tile_h / face_h) * (tile_w / face_w);  // faces per tile
    constexpr uint32_t block_w = 128;                                      // BlockW
    constexpr uint32_t tiles_per_block = block_w / tile_w;                 // BlockWt (BlockHt = 1)
    constexpr uint32_t elem_bytes = input_block_bytes / block_w;           // input element size
    constexpr uint32_t block_capacity = tile_h * block_w;                  // 4096 elems per block
    constexpr auto src_args = TensorAccessorArgs<7>();

    const auto src = TensorAccessor(src_args, src_addr);
    Noc noc;
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_scaler_obj(cb_scaler);

    // Fill the reduce scaler tile: zero, then 1.0 in row 0 of each face (reduce MAX layout).
    cb_scaler_obj.reserve_back(1);
    CoreLocalMem<volatile uint32_t> sc(cb_scaler_obj.get_write_ptr());
    noc.async_write_zeros(cb_scaler_obj, get_tile_size(cb_scaler), {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();

    for (uint32_t f = 0; f < num_faces; ++f) {
        for (uint32_t j = 0; j < face_w; ++j) {  // row 0 of the face
            sc[f * face_elems + j] = ONE_F32_BITS;
        }
    }
    cb_scaler_obj.push_back(1);

    // Stream 128-element blocks into tile_h-block batches, one bank-contiguous run per NoC read.
    const uint32_t end_row = start_row + num_rows;
    uint32_t current_row = start_row;
    uint32_t current_col = 0;  // element offset within the current row
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        cb_in_obj.reserve_back(tiles_per_block);
        uint32_t filled = 0;
        while (filled < block_capacity && current_row < end_row) {
            uint32_t space_in_block = block_capacity - filled;
            uint32_t space_in_row = width - current_col;
            uint32_t run = space_in_row < space_in_block ? space_in_row : space_in_block;
            noc.async_read(
                src,
                cb_in_obj,
                run * elem_bytes,
                {.page_id = current_row, .offset_bytes = current_col * elem_bytes},
                {.offset_bytes = filled * elem_bytes});
            filled += run;
            current_col += run;
            if (current_col >= width) {  // row consumed -> next row
                current_col = 0;
                ++current_row;
            }
        }
        // The final block may be partial (rows exhausted); its tail slots stay stale and the
        // padding-oblivious compute never has them written back by the writer.
        noc.async_read_barrier();
        cb_in_obj.push_back(tiles_per_block);
    }
}
