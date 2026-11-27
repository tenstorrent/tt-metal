// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gather_row_block (NCRISC, NoC0): face-row pieces of the TILE input land directly at their row-major
// positions in cb_row_block. Raw NoC reads: no kernel-lib helper gathers face rows of TILE pages into
// sticks (read_sticks_for_tilize reads whole sticks; the local-copy helpers are L1->L1 only). A
// gather_sticks_from_tiles helper, the producer twin of write_sticks_after_untilize, would close the gap.
//
// One read barrier per block; nothing touches the values, so all 32 bits survive by construction.

#include "api/dataflow/dataflow_api.h"
#include "g6_untilize_work_split.hpp"

void kernel_main() {
    constexpr uint32_t cb_row_block = get_compile_time_arg_val(0);
    constexpr uint32_t tensor_rows = get_compile_time_arg_val(1);
    constexpr uint32_t num_cores = get_compile_time_arg_val(2);
    constexpr uint32_t rect_w = get_compile_time_arg_val(3);
    constexpr uint32_t rows_per_image = get_compile_time_arg_val(4);
    constexpr uint32_t row_elems = get_compile_time_arg_val(5);
    constexpr uint32_t tile_rows_per_image = get_compile_time_arg_val(6);
    constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(7);
    constexpr uint32_t block_tile_cols = get_compile_time_arg_val(8);
    constexpr auto input_args = TensorAccessorArgs<9>();

    // Tile/face geometry of the CB, which the host formats as one tile of the input.
    constexpr uint32_t tile_h = unpack_tile_r_dim[cb_row_block];
    constexpr uint32_t tile_w = unpack_tile_c_dim[cb_row_block];
    constexpr uint32_t face_h = unpack_tile_face_r_dim[cb_row_block];
    constexpr uint32_t faces_per_tile_row = unpack_num_faces_c_dim[cb_row_block];
    constexpr uint32_t face_w = tile_w / faces_per_tile_row;
    constexpr uint32_t elem_bytes = get_tile_size(cb_row_block) / get_tile_hw(cb_row_block);
    constexpr uint32_t face_row_bytes = face_w * elem_bytes;
    constexpr uint32_t face_bytes = face_h * face_row_bytes;
    constexpr uint32_t face_band_bytes = faces_per_tile_row * face_bytes;
    // Row stride inside a slot: the writer helper's padded row (block_tile_cols tile rows).
    constexpr uint32_t tile_row_bytes = tile_w * elem_bytes;
    constexpr uint32_t padded_chunk_row_bytes = block_tile_cols * tile_row_bytes;

    constexpr uint32_t num_col_blocks = tensor_tile_cols / block_tile_cols;
    static_assert(num_col_blocks * block_tile_cols == tensor_tile_cols, "block_tile_cols must divide the tile cols");
    static_assert(face_row_bytes % 64 == 0, "face-row pieces must keep the BH 64-B DRAM read alignment");

    const g6_untilize::CoreRows mine = g6_untilize::core_rows<tensor_rows, num_cores, rect_w>();
    const auto input = TensorAccessor(input_args, get_common_arg_val<uint32_t>(0));

    for (uint32_t col_block_idx = 0; col_block_idx < num_col_blocks; ++col_block_idx) {
        const uint32_t tile_col0 = col_block_idx * block_tile_cols;
        for (uint32_t block_row = 0; block_row < mine.count; block_row += tile_h) {
            const uint32_t block_rows = (mine.count - block_row < tile_h) ? mine.count - block_row : tile_h;
            const uint32_t g0 = mine.start + block_row;
            const uint32_t b0 = g0 / rows_per_image;
            const uint32_t r0 = g0 - b0 * rows_per_image;

            uint32_t l1_block;
            {
                DeviceZoneScopedN("G6U-gather-issue");
                cb_reserve_back(cb_row_block, block_tile_cols);
                l1_block = get_write_ptr(cb_row_block);

                // Tile-column outer, rows inner: consecutive pieces come from the same source tile.
                for (uint32_t c = 0; c < block_tile_cols; ++c) {
                    const uint32_t tile_col = tile_col0 + c;
                    uint32_t valid_faces = faces_per_tile_row;
                    while (valid_faces > 1 && tile_col * tile_w + (valid_faces - 1) * face_w >= row_elems) {
                        --valid_faces;  // faces of pure tile padding are never read
                    }

                    uint32_t b = b0, r = r0;
                    uint32_t cur_tile_row = 0xFFFFFFFF;
                    uint64_t tile_noc_addr = 0;
                    uint32_t l1_dst = l1_block + c * tile_row_bytes;
                    for (uint32_t j = 0; j < block_rows; ++j) {
                        const uint32_t tile_row = b * tile_rows_per_image + r / tile_h;
                        if (tile_row != cur_tile_row) {
                            cur_tile_row = tile_row;
                            tile_noc_addr = input.get_noc_addr(tile_row * tensor_tile_cols + tile_col);
                        }
                        const uint32_t in_tile_r = r % tile_h;
                        uint64_t src = tile_noc_addr + (in_tile_r / face_h) * face_band_bytes +
                                       (in_tile_r % face_h) * face_row_bytes;
                        uint32_t dst = l1_dst;
                        for (uint32_t f = 0; f < valid_faces; ++f) {
                            noc_async_read_one_packet(src, dst, face_row_bytes);
                            src += face_bytes;
                            dst += face_row_bytes;
                        }
                        l1_dst += padded_chunk_row_bytes;
                        if (++r == rows_per_image) {
                            r = 0;
                            ++b;
                        }
                    }
                }
            }
            {
                DeviceZoneScopedN("G6U-gather-wait");
                noc_async_read_barrier();
            }
            {
                DeviceZoneScopedN("G6U-gather-push");
                cb_push_back(cb_row_block, block_tile_cols);
            }
        }
    }
}
