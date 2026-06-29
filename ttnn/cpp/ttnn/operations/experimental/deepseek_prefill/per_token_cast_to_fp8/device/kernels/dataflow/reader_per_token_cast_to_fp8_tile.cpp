// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for per_token_cast_to_fp8, TILE-layout input. Fills the reduce scaler tile once, then reads
// whole input tiles by tile index. For row-tile `rt` and 128-wide column-block `cb`, the block is the
// tiles_per_block tiles [rt*num_w_tiles + cb*tiles_per_block, +tiles_per_block); reading them straight
// into cb_in needs no on-core tilize. Blocks are emitted row-tile-major, column-block-minor so the
// writer can flush one row-tile's scale rows at a time.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t block_offset = get_arg_val<uint32_t>(1);          // first global block of this core
    uint32_t num_blocks = get_arg_val<uint32_t>(2);            // blocks owned by this core
    uint32_t num_w_tiles = get_arg_val<uint32_t>(3);           // input tiles across the row (H / tile_w)
    uint32_t scale_blocks_per_row = get_arg_val<uint32_t>(4);  // 128-wide column-blocks per row (H / 128)

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);
    constexpr uint32_t tile_w = get_compile_time_arg_val(3);
    constexpr uint32_t face_h = get_compile_time_arg_val(4);
    constexpr uint32_t face_w = get_compile_time_arg_val(5);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(6);
    constexpr uint32_t ONE_F32_BITS = 0x3f800000u;  // 1.0f
    constexpr uint32_t face_elems = face_h * face_w;
    constexpr uint32_t num_faces = (tile_h / face_h) * (tile_w / face_w);
    constexpr auto src_args = TensorAccessorArgs<7>();

    const auto src = TensorAccessor(src_args, src_addr);
    Noc noc;
    CircularBuffer cb_in_obj(cb_in);
    CircularBuffer cb_scaler_obj(cb_scaler);

    const uint32_t in_tile_bytes = get_tile_size(cb_in);

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

    // Each global block g maps to (row-tile g/spr, column-block g%spr); its tiles_per_block input tiles
    // are contiguous in tile-index space at rt*num_w_tiles + cb*tiles_per_block.
    const uint32_t end_block = block_offset + num_blocks;
    for (uint32_t g = block_offset; g < end_block; ++g) {
        const uint32_t rt = g / scale_blocks_per_row;
        const uint32_t cb = g % scale_blocks_per_row;
        const uint32_t block_base = rt * num_w_tiles + cb * tiles_per_block;
        cb_in_obj.reserve_back(tiles_per_block);
        for (uint32_t k = 0; k < tiles_per_block; ++k) {
            noc.async_read(
                src,
                cb_in_obj,
                in_tile_bytes,
                {.page_id = block_base + k, .offset_bytes = 0},
                {.offset_bytes = k * in_tile_bytes});
        }
        noc.async_read_barrier();
        cb_in_obj.push_back(tiles_per_block);
    }
}
