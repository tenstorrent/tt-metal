// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_pad regime N (new_storage) reader, NCRISC / NoC0.
//
// Per block of block_tiles output tiles: slot i of cb_work_tiles <-> block tile i. Every
// input-backed tile is read whole into its slot (one barrier per block); then every boundary
// slot gets c over the part that is x's tile padding but a logical output position. Fill-only
// slots stay untouched (the writer sources those from its fill page). The push is always the
// nominal block_tiles so the ring stays aligned on a ragged last block.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "g6_pad_tile_walk.hpp"

namespace {

constexpr uint32_t FACE_DIM = 16;

// c over rows [row_begin, 32) x all columns of one tile.
template <uint32_t elem_bytes>
FORCE_INLINE void mask_rows(uint32_t tile_addr, uint32_t row_begin, uint32_t fill_bits) {
    constexpr uint32_t face_row_bytes = FACE_DIM * elem_bytes;
    constexpr uint32_t face_bytes = FACE_DIM * face_row_bytes;
#pragma GCC unroll 4
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t face_row_base = (face >> 1) * FACE_DIM;
        if (row_begin >= face_row_base + FACE_DIM) {
            continue;
        }
        const uint32_t lo = row_begin > face_row_base ? row_begin - face_row_base : 0;
        dataflow_kernel_lib::fill_l1_range<elem_bytes>(
            tile_addr + face * face_bytes + lo * face_row_bytes, (FACE_DIM - lo) * face_row_bytes, fill_bits);
    }
}

// c over columns [col_begin, 32) x all rows of one tile.
template <uint32_t elem_bytes>
FORCE_INLINE void mask_cols(uint32_t tile_addr, uint32_t col_begin, uint32_t fill_bits) {
    constexpr uint32_t face_row_bytes = FACE_DIM * elem_bytes;
    constexpr uint32_t face_bytes = FACE_DIM * face_row_bytes;
#pragma GCC unroll 4
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t face_col_base = (face & 1) * FACE_DIM;
        if (col_begin >= face_col_base + FACE_DIM) {
            continue;
        }
        const uint32_t lo = col_begin > face_col_base ? col_begin - face_col_base : 0;
        const uint32_t face_addr = tile_addr + face * face_bytes;
        if (lo == 0) {
            dataflow_kernel_lib::fill_l1_range<elem_bytes>(face_addr, face_bytes, fill_bits);
            continue;
        }
        for (uint32_t r = 0; r < FACE_DIM; ++r) {
            dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                face_addr + r * face_row_bytes + lo * elem_bytes, (FACE_DIM - lo) * elem_bytes, fill_bits);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_work_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t out_lead1 = get_compile_time_arg_val(4);
    constexpr uint32_t in_lead0 = get_compile_time_arg_val(5);
    constexpr uint32_t in_lead1 = get_compile_time_arg_val(6);
    constexpr uint32_t out_row_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t out_col_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t in_row_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t in_col_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t in_row_rem = get_compile_time_arg_val(11);
    constexpr uint32_t in_col_rem = get_compile_time_arg_val(12);
    constexpr bool row_mask_on = get_compile_time_arg_val(13) != 0;
    constexpr bool col_mask_on = get_compile_time_arg_val(14) != 0;
    constexpr auto x_args = TensorAccessorArgs<15>();

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t fill_bits = get_common_arg_val<uint32_t>(1);
    const uint32_t core_work_begin = get_arg_val<uint32_t>(0);
    const uint32_t core_work_tiles = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);

    const auto x_accessor = TensorAccessor(x_args, x_addr, tile_bytes);

    using Walker =
        OutputTileWalker<out_lead1, in_lead0, in_lead1, out_row_tiles, out_col_tiles, in_row_tiles, in_col_tiles>;
    Walker walker;
    walker.init(core_work_begin);

    uint32_t remaining = core_work_tiles;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t extent = remaining < block_tiles ? remaining : block_tiles;
        const Walker block_start = walker;
        uint32_t slot_base;
        {
            DeviceZoneScopedN("G6PAD_LOAD");
            {
                DeviceZoneScopedN("G6PAD_RESERVE_WAIT");
                cb_reserve_back(cb_work_tiles, block_tiles);
            }
            slot_base = get_write_ptr(cb_work_tiles);
            for (uint32_t i = 0; i < extent; ++i) {
                if (walker.has_input()) {
                    noc_async_read_page(walker.in_page(), x_accessor, slot_base + i * tile_bytes);
                }
                walker.advance();
            }
            {
                DeviceZoneScopedN("G6PAD_READ_WAIT");
                noc_async_read_barrier();
            }
        }
        {
            DeviceZoneScopedN("G6PAD_ASSEMBLE");
            if constexpr (row_mask_on || col_mask_on) {
                // Strictly after the barrier: an in-flight read would overwrite the mask.
                Walker tile = block_start;
                for (uint32_t i = 0; i < extent; ++i) {
                    if (tile.has_input()) {
                        const uint32_t tile_addr = slot_base + i * tile_bytes;
                        if constexpr (row_mask_on) {
                            if (tile.last_in_row()) {
                                mask_rows<elem_bytes>(tile_addr, in_row_rem, fill_bits);
                            }
                        }
                        if constexpr (col_mask_on) {
                            if (tile.last_in_col()) {
                                mask_cols<elem_bytes>(tile_addr, in_col_rem, fill_bits);
                            }
                        }
                    }
                    tile.advance();
                }
            }
            cb_push_back(cb_work_tiles, block_tiles);
        }
        remaining -= extent;
    }
}
