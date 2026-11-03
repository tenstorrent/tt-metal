// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"

// Baseline reader: one packed tile (c0, face order) and its raw E8M0 scale
// bytes (c1, one page) per iteration. ROW_MAJOR input is staged in c2 and
// rearranged into faces so the compute kernel always sees tile order.
void kernel_main() {
    constexpr uint32_t height = get_compile_time_arg_val(0);
    constexpr uint32_t width = get_compile_time_arg_val(1);
    constexpr uint32_t word_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t width_tiles = get_compile_time_arg_val(3);
    constexpr bool tiled = get_compile_time_arg_val(4);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t scale_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t staging_stride = get_compile_time_arg_val(7);
    constexpr auto packed_args = TensorAccessorArgs<8>();
    constexpr auto scale_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
    const auto packed = TensorAccessor(packed_args, get_arg_val<uint32_t>(0), page_bytes);
    const auto scales = TensorAccessor(scale_args, get_arg_val<uint32_t>(1), scale_bytes);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    CircularBuffer data_cb(0), scale_cb(1), scratch(2);
    Noc noc;
    constexpr uint32_t face_height = height / 2;
    constexpr uint32_t face_width = width / 2;
    constexpr uint32_t row_bytes = width * word_bytes;
    for (uint32_t tile = start; tile < start + count; ++tile) {
        data_cb.reserve_back(1);
        scale_cb.reserve_back(1);
        noc.async_read(scales, scale_cb, scale_bytes, {.page_id = tile}, {});
        if constexpr (tiled) {
            noc.async_read(packed, data_cb, height * row_bytes, {.page_id = tile}, {});
            noc.async_read_barrier();
        } else {
            // Read whole tile rows before the barrier, then arrange into faces.
            const uint32_t first_row = (tile / width_tiles) * height;
            const uint32_t col_bytes = (tile % width_tiles) * row_bytes;
            // Blackhole DRAM reads require source/destination low address bits
            // to match. UINT8 tile rows are only 32 bytes: pad the staging stride
            // and carry the source column alignment into the local address.
            const uint32_t staging_offset = col_bytes & (staging_stride - 1);
            for (uint32_t row = 0; row < height; ++row) {
                noc.async_read(
                    packed,
                    scratch,
                    row_bytes,
                    {.page_id = first_row + row, .offset_bytes = col_bytes},
                    {.offset_bytes = row * staging_stride + staging_offset});
            }
            noc.async_read_barrier();
            const auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch.get_write_ptr() + staging_offset);
            auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_cb.get_write_ptr());
            constexpr uint32_t face_row_words = face_width * word_bytes / sizeof(uint32_t);
            for (uint32_t face = 0; face < 4; ++face) {
                for (uint32_t row = 0; row < face_height; ++row) {
                    const uint32_t src_row = (face / 2) * face_height + row;
                    for (uint32_t col = 0; col < face_row_words; ++col) {
                        dst[(face * face_height + row) * face_row_words + col] =
                            src[src_row * staging_stride / sizeof(uint32_t) + (face % 2) * face_row_words + col];
                    }
                }
            }
        }
        data_cb.push_back(1);
        scale_cb.push_back(1);
    }
}
