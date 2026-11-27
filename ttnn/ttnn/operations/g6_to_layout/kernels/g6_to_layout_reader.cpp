// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_to_layout reader (NCRISC / NoC0): load_live_tiles.
//
// Walks this core's unit range [unit_begin, unit_end) (unit = column group j x tile_row,
// column-group-major) in blocks of block_height_tiles x block_width_tiles. Per block: one
// cb_reserve_back, every read of the block in flight, one barrier, one cb_push_back of the
// nominal block_tiles pages. Raw NoC: no kernel_lib helper reads interleaved TILE pages with
// face-level live extents.
//
// Only live faces are read, and of each live face only its live-row prefix; unread bytes of
// the slot are stale and map only to positions the writer never writes.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;

template <uint32_t face_row_bytes, uint32_t face_bytes, uint32_t tile_bytes>
FORCE_INLINE void read_live_tile(uint64_t src, uint32_t dst, uint32_t live_rows, uint32_t live_cols) {
    const uint32_t top_rows = live_rows < FACE_DIM ? live_rows : FACE_DIM;
    const uint32_t bottom_rows = live_rows - top_rows;
    if (live_cols > FACE_DIM) {
        if (bottom_rows == FACE_DIM) {
            noc_async_read(src, dst, tile_bytes);
        } else if (bottom_rows > 0) {
            // Faces 0 and 1 whole and face 2's prefix are contiguous; face 3's prefix follows.
            noc_async_read(src, dst, 2 * face_bytes + bottom_rows * face_row_bytes);
            noc_async_read(src + 3 * face_bytes, dst + 3 * face_bytes, bottom_rows * face_row_bytes);
        } else if (top_rows == FACE_DIM) {
            noc_async_read(src, dst, 2 * face_bytes);
        } else {
            noc_async_read(src, dst, top_rows * face_row_bytes);
            noc_async_read(src + face_bytes, dst + face_bytes, top_rows * face_row_bytes);
        }
    } else {
        noc_async_read(src, dst, top_rows * face_row_bytes);
        if (bottom_rows > 0) {
            noc_async_read(src + 2 * face_bytes, dst + 2 * face_bytes, bottom_rows * face_row_bytes);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_tiles_in = get_compile_time_arg_val(0);
    constexpr uint32_t block_height_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_tile_rows = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(4);
    constexpr uint32_t image_tile_rows = get_compile_time_arg_val(5);
    constexpr uint32_t height = get_compile_time_arg_val(6);
    constexpr uint32_t width = get_compile_time_arg_val(7);
    constexpr uint32_t col_group_width_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(9);
    constexpr auto input_args = TensorAccessorArgs<10>();

    constexpr uint32_t datum_bytes = tile_bytes / (TILE_DIM * TILE_DIM);
    constexpr uint32_t face_row_bytes = FACE_DIM * datum_bytes;
    constexpr uint32_t face_bytes = FACE_DIM * face_row_bytes;
    constexpr uint32_t block_tiles = block_height_tiles * block_width_tiles;

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t unit_begin = get_arg_val<uint32_t>(0);
    const uint32_t unit_end = get_arg_val<uint32_t>(1);

    const auto input = TensorAccessor(input_args, input_addr, tile_bytes);

    DeviceZoneScopedN("g6_reader");
    uint32_t unit = unit_begin;
    while (unit < unit_end) {
        const uint32_t group = unit / tensor_tile_rows;
        const uint32_t first_tile_row = unit - group * tensor_tile_rows;
        uint32_t rows = block_height_tiles;
        if (tensor_tile_rows - first_tile_row < rows) {
            rows = tensor_tile_rows - first_tile_row;
        }
        if (unit_end - unit < rows) {
            rows = unit_end - unit;
        }
        const uint32_t col_begin = group * col_group_width_tiles;
        const uint32_t col_end =
            col_begin + col_group_width_tiles < tensor_tile_cols ? col_begin + col_group_width_tiles : tensor_tile_cols;

        for (uint32_t col = col_begin; col < col_end; col += block_width_tiles) {
            const uint32_t cols = col_end - col < block_width_tiles ? col_end - col : block_width_tiles;

            cb_reserve_back(cb_tiles_in, block_tiles);
            const uint32_t block_base = get_write_ptr(cb_tiles_in);
            uint32_t tile_row = first_tile_row;
            uint32_t ht = first_tile_row % image_tile_rows;
            for (uint32_t t = 0; t < rows; ++t, ++tile_row) {
                const uint32_t rows_left = height - ht * TILE_DIM;
                const uint32_t live_rows = rows_left < TILE_DIM ? rows_left : TILE_DIM;
                const uint32_t row_page = tile_row * tensor_tile_cols + col;
                const uint32_t row_dst = block_base + t * block_width_tiles * tile_bytes;
                for (uint32_t c = 0; c < cols; ++c) {
                    const uint32_t cols_left = width - (col + c) * TILE_DIM;
                    const uint32_t live_cols = cols_left < TILE_DIM ? cols_left : TILE_DIM;
                    read_live_tile<face_row_bytes, face_bytes, tile_bytes>(
                        input.get_noc_addr(row_page + c), row_dst + c * tile_bytes, live_rows, live_cols);
                }
                if (++ht == image_tile_rows) {
                    ht = 0;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_tiles_in, block_tiles);
        }
        unit += rows;
    }
}
