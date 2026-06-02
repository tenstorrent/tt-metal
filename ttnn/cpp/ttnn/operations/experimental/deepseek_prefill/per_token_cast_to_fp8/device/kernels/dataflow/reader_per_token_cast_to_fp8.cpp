// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for per_token_cast_to_fp8. Fills the reduce scaler tile (1.0 in row 0 of each face, the
// reduce-LLK MAX layout) once, then reads the fp32/bf16 input column-block-major: per (tile-row,
// col-block), one input tile (1024 elements) per row -> cb_in (one tile per page). FIFO consume.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/experimental/minimal_matmul/device/kernels/matmul_dataflow_common.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(1);
    uint32_t num_col_blocks = get_arg_val<uint32_t>(2);
    uint32_t start_tile_row = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t col_block_bytes = get_compile_time_arg_val(1);  // 1024 * elem_size
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    // Tile / face dims from the tensor's tile spec (32x32 / 16x16 by default; tiny tiles supported).
    constexpr uint32_t tile_h = get_compile_time_arg_val(3);
    constexpr uint32_t tile_w = get_compile_time_arg_val(4);
    constexpr uint32_t face_h = get_compile_time_arg_val(5);
    constexpr uint32_t face_w = get_compile_time_arg_val(6);
    constexpr uint32_t ONE_F32_BITS = 0x3f800000u;  // 1.0f
    constexpr uint32_t face_elems = face_h * face_w;                       // fp32 elements per face
    constexpr uint32_t num_faces = (tile_h / face_h) * (tile_w / face_w);  // faces per tile
    constexpr auto src_args = TensorAccessorArgs<7>();

    const auto src = TensorAccessor(src_args, src_addr);

    // Fill the reduce scaler tile: zero, then 1.0 in row 0 of each face (reduce MAX layout).
    cb_reserve_back(cb_scaler, 1);
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
    fill_zeros_async(get_write_ptr(cb_scaler), get_tile_size(cb_scaler));

    for (uint32_t f = 0; f < num_faces; ++f) {
        for (uint32_t j = 0; j < face_w; ++j) {  // row 0 of the face
            sc[f * face_elems + j] = ONE_F32_BITS;
        }
    }
    cb_push_back(cb_scaler, 1);

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        for (uint32_t c = 0; c < num_col_blocks; ++c) {
            cb_reserve_back(cb_in, tile_h);
            uint32_t l1 = get_write_ptr(cb_in);
            uint32_t col_offset_bytes = c * col_block_bytes;
            for (uint32_t s = 0; s < tile_h; ++s) {
                uint32_t page_id = (start_tile_row + tr) * tile_h + s;
                noc_async_read(src.get_noc_addr(page_id) + col_offset_bytes, l1, col_block_bytes);
                l1 += col_block_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, tile_h);
        }
    }
}
