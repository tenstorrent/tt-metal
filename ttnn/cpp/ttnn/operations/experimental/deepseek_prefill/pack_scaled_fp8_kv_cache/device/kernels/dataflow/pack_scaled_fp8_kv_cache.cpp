// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    const uint32_t latent_addr = get_arg_val<uint32_t>(0);
    const uint32_t scale_addr = get_arg_val<uint32_t>(1);
    const uint32_t rope_addr = get_arg_val<uint32_t>(2);
    const uint32_t output_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_row = get_arg_val<uint32_t>(4);
    const uint32_t num_rows = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_scratch_id = get_compile_time_arg_val(0);
    constexpr uint32_t latent_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t scale_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t rope_bytes = get_compile_time_arg_val(3);
    constexpr bool rope_tiled = get_compile_time_arg_val(4);
    constexpr uint32_t cb_rope_tiles_id = get_compile_time_arg_val(5);
    constexpr uint32_t rope_rows_per_group = get_compile_time_arg_val(6);
    constexpr uint32_t rope_tile_rows_per_group = get_compile_time_arg_val(7);
    constexpr uint32_t scale_offset = latent_bytes;
    constexpr uint32_t rope_offset = latent_bytes + scale_bytes;

    constexpr auto latent_args = TensorAccessorArgs<8>();
    constexpr auto scale_args = TensorAccessorArgs<latent_args.next_compile_time_args_offset()>();
    constexpr auto rope_args = TensorAccessorArgs<scale_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<rope_args.next_compile_time_args_offset()>();
    const auto latent = TensorAccessor(latent_args, latent_addr);
    const auto scales = TensorAccessor(scale_args, scale_addr);
    const auto rope = TensorAccessor(rope_args, rope_addr);
    const auto output = TensorAccessor(output_args, output_addr);

    Noc noc;
    CircularBuffer scratch(cb_scratch_id);
    CircularBuffer rope_tiles(cb_rope_tiles_id);
    uint32_t cached_rope_tile_row = 0xffffffffu;
    for (uint32_t row = start_row; row < start_row + num_rows; ++row) {
        scratch.reserve_back(1);
        noc.async_read(latent, scratch, latent_bytes, {.page_id = row}, {.offset_bytes = 0});
        noc.async_read_barrier();
        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(scratch),
            output,
            latent_bytes,
            {.offset_bytes = 0},
            {.page_id = row});
        noc.async_write_barrier();

        noc.async_read(scales, scratch, scale_bytes, {.page_id = row}, {.offset_bytes = 0});
        noc.async_read_barrier();
        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(scratch),
            output,
            scale_bytes,
            {.offset_bytes = 0},
            {.page_id = row, .offset_bytes = scale_offset});
        noc.async_write_barrier();

        if constexpr (rope_tiled) {
            constexpr uint32_t tile_bytes = 32 * 32 * sizeof(uint16_t);
            constexpr uint32_t face_bytes = 16 * 16 * sizeof(uint16_t);
            constexpr uint32_t face_line_words = 16 * sizeof(uint16_t) / sizeof(uint32_t);
            const uint32_t group = row / rope_rows_per_group;
            const uint32_t row_in_group = row % rope_rows_per_group;
            const uint32_t tile_row = group * rope_tile_rows_per_group + row_in_group / 32;
            if (tile_row != cached_rope_tile_row) {
                if (cached_rope_tile_row != 0xffffffffu) {
                    rope_tiles.pop_front(1);
                }
                rope_tiles.reserve_back(1);
                const uint32_t tile_ptr = rope_tiles.get_write_ptr();
                noc.async_read(rope, CoreLocalMem<uint32_t>(tile_ptr), tile_bytes, {.page_id = tile_row * 2}, {});
                noc.async_read(
                    rope, CoreLocalMem<uint32_t>(tile_ptr + tile_bytes), tile_bytes, {.page_id = tile_row * 2 + 1}, {});
                noc.async_read_barrier();
                rope_tiles.push_back(1);
                rope_tiles.wait_front(1);
                cached_rope_tile_row = tile_row;
            }
            const uint32_t row_in_tile = row_in_group % 32;
            const uint32_t face_row = row_in_tile % 16;
            // BF16 tiles store four 16x16 faces. Copy the four face lines in logical
            // column order (two faces from each of the two 32-column tiles).
            const uint32_t face_pair_offset = (row_in_tile / 16) * 2 * face_bytes + face_row * 16 * sizeof(uint16_t);
            const uint32_t tile_ptr = rope_tiles.get_read_ptr();
            auto* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch.get_write_ptr());
            for (uint32_t tile_col = 0; tile_col < 2; ++tile_col) {
                for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                    auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        tile_ptr + tile_col * tile_bytes + face_pair_offset + face_col * face_bytes);
                    for (uint32_t word = 0; word < face_line_words; ++word) {
                        destination[(tile_col * 2 + face_col) * face_line_words + word] = source[word];
                    }
                }
            }
        } else {
            noc.async_read(rope, scratch, rope_bytes, {.page_id = row}, {.offset_bytes = 0});
            noc.async_read_barrier();
        }
        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(scratch),
            output,
            rope_bytes,
            {.offset_bytes = 0},
            {.page_id = row, .offset_bytes = rope_offset});
        noc.async_write_barrier();
        scratch.push_back(1);
        scratch.pop_front(1);
    }
    if constexpr (rope_tiled) {
        if (cached_rope_tile_row != 0xffffffffu) {
            rope_tiles.pop_front(1);
        }
    }
}
