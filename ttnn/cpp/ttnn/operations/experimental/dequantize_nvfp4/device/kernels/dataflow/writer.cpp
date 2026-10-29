// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    constexpr uint32_t height = get_compile_time_arg_val(0);
    constexpr uint32_t width = get_compile_time_arg_val(1);
    constexpr uint32_t element_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t width_tiles = get_compile_time_arg_val(3);
    constexpr bool tiled = get_compile_time_arg_val(4);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(5);
    constexpr auto output_args = TensorAccessorArgs<6>();
    constexpr uint32_t transpose_args_offset = output_args.next_compile_time_args_offset();
    constexpr bool transpose_output = get_compile_time_arg_val(transpose_args_offset);
    constexpr uint32_t original_height_tiles = get_compile_time_arg_val(transpose_args_offset + 1);
    constexpr uint32_t original_width_tiles = get_compile_time_arg_val(transpose_args_offset + 2);
    static_assert(!transpose_output || tiled, "Transposed NVFP4 output requires TILE layout");
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(0), page_bytes);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    CircularBuffer output_cb(16), scratch(17);
    Noc noc;
    constexpr uint32_t face_height = height / 2;
    constexpr uint32_t face_width = width / 2;
    constexpr uint32_t row_bytes = width * element_bytes;
    for (uint32_t tile = start; tile < start + count; ++tile) {
        output_cb.wait_front(1);
        if constexpr (tiled) {
            uint32_t output_tile = tile;
            if constexpr (transpose_output) {
                constexpr uint32_t matrix_tiles = original_height_tiles * original_width_tiles;
                const uint32_t batch = tile / matrix_tiles;
                const uint32_t matrix_tile = tile % matrix_tiles;
                const uint32_t row = matrix_tile / original_width_tiles;
                const uint32_t col = matrix_tile % original_width_tiles;
                output_tile = batch * matrix_tiles + col * original_height_tiles + row;
            }
            noc.async_write(output_cb, output, height * row_bytes, {}, {.page_id = output_tile});
        } else {
            const auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_cb.get_read_ptr());
            auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch.get_write_ptr());
            constexpr uint32_t face_row_words = face_width * element_bytes / sizeof(uint32_t);
            for (uint32_t face = 0; face < 4; ++face) {
                for (uint32_t row = 0; row < face_height; ++row) {
                    const uint32_t dst_row = (face / 2) * face_height + row;
                    for (uint32_t col = 0; col < face_row_words; ++col) {
                        dst[dst_row * row_bytes / sizeof(uint32_t) + (face % 2) * face_row_words + col] =
                            src[(face * face_height + row) * face_row_words + col];
                    }
                }
            }
            const uint32_t first_row = (tile / width_tiles) * height;
            const uint32_t col_bytes = (tile % width_tiles) * row_bytes;
            for (uint32_t row = 0; row < height; ++row) {
                // Scratch is a raw, non-FIFO staging buffer; both pointers stay at its base.
                noc.async_write(
                    scratch,
                    output,
                    row_bytes,
                    {.offset_bytes = row * row_bytes},
                    {.page_id = first_row + row, .offset_bytes = col_bytes});
            }
        }
        noc.async_write_barrier();
        output_cb.pop_front(1);
    }
}
