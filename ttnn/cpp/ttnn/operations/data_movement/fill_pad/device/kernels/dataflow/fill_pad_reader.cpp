// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Reads border tiles from the tensor (DRAM interleaved or sharded) into the
 * double-buffered cb_tile_in. The writer (BRISC) applies the padding-fill mask
 * in L1 and writes each tile back.
 *
 * Unified border-tile split. The host enumerates border tiles across all
 * slices into three contiguous blocks (right / bottom / corner) and gives
 * this core a per-phase (start, num) range inside each block:
 *
 *   Right phase  (num_right > 0 only when has_right_pad):
 *     for each i in [start_right, start_right + num_right):
 *       slice = i / right_slice_stride;  row = i % right_slice_stride
 *       tile_id = slice * H_tiles * W_tiles + row * W_tiles + (W_tiles - 1)
 *     where right_slice_stride = (H_tiles - 1) if has_bottom_pad else H_tiles.
 *
 *   Bottom phase (num_bottom > 0 only when has_bottom_pad):
 *     for each j in [start_bottom, start_bottom + num_bottom):
 *       slice = j / bottom_slice_stride;  col = j % bottom_slice_stride
 *       tile_id = slice * H_tiles * W_tiles + (H_tiles - 1) * W_tiles + col
 *     where bottom_slice_stride = (W_tiles - 1) if has_right_pad else W_tiles.
 *
 *   Corner phase (num_corner > 0 only when has_right_pad && has_bottom_pad):
 *     for each k in [start_corner, start_corner + num_corner):
 *       slice = k
 *       tile_id = slice * H_tiles * W_tiles + (H_tiles - 1) * W_tiles + (W_tiles - 1)
 *
 * Tile ordering across the three phases must match fill_pad_writer.cpp and
 * fill_pad_compute.cpp exactly (CBs are FIFO).
 *
 * CT args layout (common to reader and writer):
 *   [0]  W_tiles
 *   [1]  H_tiles
 *   [2]  N_slices (unused here)
 *   [3]  has_right_pad
 *   [4]  has_bottom_pad
 *   [5]  W_mod32 (unused here)
 *   [6]  H_mod32 (unused here)
 *   [7]  elem_size (2 or 4)
 *   [8]  fill_bits (unused here)
 *   [9]  CB_TILE_IN (= 0)
 *   [10+] TensorAccessorArgs
 *
 * RT args:
 *   [0]  buf_addr
 *   [1]  start_right   [2]  num_right
 *   [3]  start_bottom  [4]  num_bottom
 *   [5]  start_corner  [6]  num_corner
 */

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr uint32_t W_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t H_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t has_right_pad = get_compile_time_arg_val(3);
    constexpr uint32_t has_bottom_pad = get_compile_time_arg_val(4);
    constexpr uint32_t elem_size = get_compile_time_arg_val(7);
    constexpr uint32_t cb_tile_in_idx = get_compile_time_arg_val(9);

    // Per-phase slice strides (meaningful only when the corresponding phase is active).
    // Clamped to >= 1 so the compiler does not see a constexpr divide-by-zero in
    // the dead-code branches (when H_tiles==1 or W_tiles==1 the host sets the
    // matching num_* to 0 and the loop below never executes).
    constexpr uint32_t right_slice_stride =
        has_right_pad ? (has_bottom_pad ? ((H_tiles > 1u) ? (H_tiles - 1u) : 1u) : H_tiles) : 1u;
    constexpr uint32_t bottom_slice_stride =
        has_bottom_pad ? (has_right_pad ? ((W_tiles > 1u) ? (W_tiles - 1u) : 1u) : W_tiles) : 1u;

    constexpr uint32_t tile_bytes = get_tile_size(cb_tile_in_idx);

    const uint32_t buf_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_right = get_arg_val<uint32_t>(1);
    const uint32_t num_right = get_arg_val<uint32_t>(2);
    const uint32_t start_bottom = get_arg_val<uint32_t>(3);
    const uint32_t num_bottom = get_arg_val<uint32_t>(4);
    const uint32_t start_corner = get_arg_val<uint32_t>(5);
    const uint32_t num_corner = get_arg_val<uint32_t>(6);

    constexpr auto src_args = TensorAccessorArgs<10>();
    const auto s = TensorAccessor(src_args, buf_addr, tile_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb_tile_in(cb_tile_in_idx);

    // ---- Right phase ----
    // Maintain (slice, row) incrementally instead of dividing every iteration
    // — RV32IM division is slow. Startup division runs at most once.
    if constexpr (has_right_pad) {
        uint32_t slice = num_right ? start_right / right_slice_stride : 0u;
        uint32_t row = num_right ? start_right - slice * right_slice_stride : 0u;
        for (uint32_t i = 0; i < num_right; ++i) {
            const uint32_t tile_id = slice * H_tiles * W_tiles + row * W_tiles + (W_tiles - 1u);
            cb_tile_in.reserve_back(1);
            noc.async_read(s, cb_tile_in, tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_tile_in.push_back(1);
            ++row;
            if (row == right_slice_stride) {
                row = 0;
                ++slice;
            }
        }
    }

    // ---- Bottom phase ----
    if constexpr (has_bottom_pad) {
        uint32_t slice = num_bottom ? start_bottom / bottom_slice_stride : 0u;
        uint32_t col = num_bottom ? start_bottom - slice * bottom_slice_stride : 0u;
        for (uint32_t j = 0; j < num_bottom; ++j) {
            const uint32_t tile_id = slice * H_tiles * W_tiles + (H_tiles - 1u) * W_tiles + col;
            cb_tile_in.reserve_back(1);
            noc.async_read(s, cb_tile_in, tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_tile_in.push_back(1);
            ++col;
            if (col == bottom_slice_stride) {
                col = 0;
                ++slice;
            }
        }
    }

    // ---- Corner phase ----
    if constexpr (has_right_pad && has_bottom_pad) {
        for (uint32_t k = 0; k < num_corner; ++k) {
            const uint32_t slice = start_corner + k;
            const uint32_t tile_id = slice * H_tiles * W_tiles + (H_tiles - 1u) * W_tiles + (W_tiles - 1u);
            cb_tile_in.reserve_back(1);
            noc.async_read(s, cb_tile_in, tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_tile_in.push_back(1);
        }
    }
}
