// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast reader (NCRISC / NoC0) — block operation `load_block` (op_design.md, Block schedule).
// Streams this core's contiguous range of input tiles [start_tile, start_tile + core_tiles) into
// cb_input_tiles in blocks of `block_tiles`: block_tiles page reads in flight, ONE read barrier and
// ONE reserve/push handshake per block. A ragged last block reserves/pushes its actual extent
// (compute and writer synchronize the same per-block extent).
// No dataflow helper owns an interleaved page-range stream, so this is the raw TensorAccessor loop.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"

namespace {

// bfloat8_b output only: zero the padding columns (col >= w_tail) of a last-tile-column tile so the
// packer's shared exponent of a straddling 16-value group never sees padding values.
// Tile = 4 faces of 16x16 bf16 (face0 TL, face1 TR, face2 BL, face3 BR), row-major inside a face.
template <uint32_t w_tail>
FORCE_INLINE void zero_pad_columns(const Noc& noc, const CircularBuffer& cb, uint32_t tile_offset_bytes) {
    constexpr uint32_t face_dim = 16;
    constexpr uint32_t elem_bytes = 2;
    constexpr uint32_t face_bytes = face_dim * face_dim * elem_bytes;
    if constexpr (w_tail < face_dim) {
        // Right faces (1, 3) are entirely padding: zero them with the DM engine.
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset_bytes + 1 * face_bytes});
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset_bytes + 3 * face_bytes});
    }
    // Partial face rows: only the padding lanes of the face that straddles W.
    constexpr uint32_t col0 = w_tail % face_dim;  // first padding column inside the straddling face
    constexpr uint32_t face_a = (w_tail < face_dim) ? 0 : 1;
    constexpr uint32_t face_b = face_a + 2;
    if constexpr (col0 != 0) {
        uint32_t base = cb.get_write_ptr() + tile_offset_bytes;
        for (uint32_t face : {face_a, face_b}) {
            volatile tt_l1_ptr uint16_t* f = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + face * face_bytes);
            for (uint32_t r = 0; r < face_dim; ++r) {
                for (uint32_t c = col0; c < face_dim; ++c) {
                    f[r * face_dim + c] = 0;
                }
            }
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t in_page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t mask_pad_columns = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(4);
    constexpr uint32_t w_tail = get_compile_time_arg_val(5);
    constexpr auto input_args = TensorAccessorArgs<6>();

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(0);
    const uint32_t core_tiles = get_arg_val<uint32_t>(1);

    const auto input_accessor = TensorAccessor(input_args, input_addr, in_page_bytes);
    [[maybe_unused]] Noc noc;
    CircularBuffer cb_in(cb_input_tiles);

    const uint32_t end_tile = start_tile + core_tiles;
    for (uint32_t block_start = start_tile; block_start < end_tile; block_start += block_tiles) {
        DeviceZoneScopedN("READ_BLOCK");
        const uint32_t block_extent = (end_tile - block_start < block_tiles) ? (end_tile - block_start) : block_tiles;
        {
            DeviceZoneScopedN("READ_WAIT_SPACE");
            cb_reserve_back(cb_input_tiles, block_extent);
        }
        uint32_t l1_addr = get_write_ptr(cb_input_tiles);
        for (uint32_t t = 0; t < block_extent; ++t) {
            noc_async_read_page(block_start + t, input_accessor, l1_addr);
            l1_addr += in_page_bytes;
        }
        noc_async_read_barrier();

        if constexpr (mask_pad_columns) {
            // Last-tile-column tiles of this block: page id % tensor_tile_cols == tensor_tile_cols - 1.
            for (uint32_t t = 0; t < block_extent; ++t) {
                if ((block_start + t) % tensor_tile_cols == tensor_tile_cols - 1) {
                    zero_pad_columns<w_tail>(noc, cb_in, t * in_page_bytes);
                }
            }
            noc.write_zeros_l1_barrier();
        }

        cb_push_back(cb_input_tiles, block_extent);
    }
}
