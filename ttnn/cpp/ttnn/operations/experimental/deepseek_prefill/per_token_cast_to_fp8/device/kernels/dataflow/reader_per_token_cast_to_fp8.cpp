// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for per_token_cast_to_fp8 (group-major). Fills the reduce scaler tile once, then streams
// the core's rows as a flat sequence of 128-element groups. A "block" is tile_h consecutive groups
// (block_capacity = tile_h * 128 elements = COL_BLOCK_TILES tiles). Each bank-contiguous run (a span
// within one row) is read with a single noc_async_read, so we exploit row-major DRAM locality
// instead of hopping banks. After tilize, the block is [tile_h groups x 128] and the (unchanged,
// GROUPS_PER_BLOCK=1) compute reduces each group independently.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/experimental/minimal_matmul/device/kernels/matmul_dataflow_common.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);
    uint32_t start_row = get_arg_val<uint32_t>(2);  // absolute first row of this core's stream
    uint32_t num_rows = get_arg_val<uint32_t>(3);   // rows owned by this core
    uint32_t width = get_arg_val<uint32_t>(4);      // H (elements per row)

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t group_bytes = get_compile_time_arg_val(1);  // 128 * elem_size (one group)
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    // Tile / face dims from the tensor's tile spec.
    constexpr uint32_t tile_h = get_compile_time_arg_val(3);
    constexpr uint32_t tile_w = get_compile_time_arg_val(4);
    constexpr uint32_t face_h = get_compile_time_arg_val(5);
    constexpr uint32_t face_w = get_compile_time_arg_val(6);
    constexpr uint32_t ONE_F32_BITS = 0x3f800000u;  // 1.0f
    constexpr uint32_t face_elems = face_h * face_w;                       // fp32 elements per face
    constexpr uint32_t num_faces = (tile_h / face_h) * (tile_w / face_w);  // faces per tile
    constexpr uint32_t COL_BLOCK_ELEMS = 128;                              // column-block width = one group
    constexpr uint32_t COL_BLOCK_TILES = COL_BLOCK_ELEMS / tile_w;         // 4 tiles per block
    constexpr uint32_t elem_bytes = group_bytes / COL_BLOCK_ELEMS;         // input element size
    constexpr uint32_t block_capacity = tile_h * COL_BLOCK_ELEMS;          // 4096 elems per block
    constexpr auto src_args = TensorAccessorArgs<7>();

    const auto src = TensorAccessor(src_args, src_addr);

    // Fill the reduce scaler tile: zero, then 1.0 in row 0 of each face (reduce MAX layout).
    cb_reserve_back(cb_scaler, 1);
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
    fill_zeros_async(get_write_ptr(cb_scaler), get_tile_size(cb_scaler));
    noc_async_read_barrier();

    for (uint32_t f = 0; f < num_faces; ++f) {
        for (uint32_t j = 0; j < face_w; ++j) {  // row 0 of the face
            sc[f * face_elems + j] = ONE_F32_BITS;
        }
    }
    cb_push_back(cb_scaler, 1);

    // Stream groups into tile_h-group blocks, one bank-contiguous run per noc_async_read.
    const uint32_t end_row = start_row + num_rows;
    uint32_t current_row = start_row;
    uint32_t current_col = 0;  // element offset within the current row
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        cb_reserve_back(cb_in, COL_BLOCK_TILES);
        uint32_t l1 = get_write_ptr(cb_in);
        uint32_t filled = 0;
        while (filled < block_capacity && current_row < end_row) {
            uint32_t space_in_block = block_capacity - filled;
            uint32_t space_in_row = width - current_col;
            uint32_t run = space_in_row < space_in_block ? space_in_row : space_in_block;
            noc_async_read(src.get_noc_addr(current_row) + current_col * elem_bytes, l1, run * elem_bytes);
            filled += run;
            current_col += run;
            l1 += run * elem_bytes;
            if (current_col >= width) {  // row consumed -> next row
                current_col = 0;
                ++current_row;
            }
        }
        // The final block may be partial (rows exhausted); its tail slots stay stale and the
        // padding-oblivious compute never has them written back by the writer.
        noc_async_read_barrier();
        cb_push_back(cb_in, COL_BLOCK_TILES);
    }
}
