// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Phase 1 – Mask generation (before the main loop):
 *   Builds a "right mask" tile (if has_right_pad) and a "bottom mask" tile
 *   (if has_bottom_pad) in face layout and pushes them to their respective
 *   circular buffers. The compute kernel holds these tiles persistently
 *   (never pops them) and uses them with where_tile to apply the fill.
 *
 *   Mask encoding (same DataFormat as the input tensor):
 *     Float types  : 1.0 at padding positions, 0.0 elsewhere.
 *     Integer types: integer 1 at padding positions, 0 elsewhere.
 *
 * Phase 2 – Write-back loop:
 *   Reads masked tiles produced by the compute kernel from CB[16] and writes
 *   them back to DRAM (or sharded L1). No masking is done here.
 *
 *   Three phase loops mirror fill_pad_reader.cpp's right / bottom / corner
 *   phases, using the same per-phase (start, num) RT args so that reader,
 *   compute and writer process tiles in lock-step.
 *
 * CT args layout:
 *   [0]  W_tiles
 *   [1]  H_tiles
 *   [2]  N_slices (unused in the write loop)
 *   [3]  has_right_pad
 *   [4]  has_bottom_pad
 *   [5]  W_mod32
 *   [6]  H_mod32
 *   [7]  cb_right_mask_idx  (= 1)
 *   [8]  cb_bot_mask_idx    (= 2)
 *   [9]  cb_data_out_idx    (= 16)
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
#include "fill_pad_dataflow_common.hpp"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr uint32_t W_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t H_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t has_right_pad = get_compile_time_arg_val(3);
    constexpr uint32_t has_bottom_pad = get_compile_time_arg_val(4);
    constexpr uint32_t W_mod32 = get_compile_time_arg_val(5);
    constexpr uint32_t H_mod32 = get_compile_time_arg_val(6);
    constexpr uint32_t cb_right_mask_idx = get_compile_time_arg_val(7);
    constexpr uint32_t cb_bot_mask_idx = get_compile_time_arg_val(8);
    constexpr uint32_t cb_data_out_idx = get_compile_time_arg_val(9);

    // Per-phase slice strides (meaningful only when the corresponding phase is active).
    // Clamped to >= 1 so the compiler does not see a constexpr divide-by-zero in
    // the dead-code branches (when H_tiles==1 or W_tiles==1 the host sets the
    // matching num_* to 0 and the loop below never executes).
    constexpr uint32_t right_slice_stride =
        has_right_pad ? (has_bottom_pad ? ((H_tiles > 1u) ? (H_tiles - 1u) : 1u) : H_tiles) : 1u;
    constexpr uint32_t bottom_slice_stride =
        has_bottom_pad ? (has_right_pad ? ((W_tiles > 1u) ? (W_tiles - 1u) : 1u) : W_tiles) : 1u;

    constexpr uint32_t tile_bytes = get_tile_size(cb_data_out_idx);

    const uint32_t buf_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_right = get_arg_val<uint32_t>(1);
    const uint32_t num_right = get_arg_val<uint32_t>(2);
    const uint32_t start_bottom = get_arg_val<uint32_t>(3);
    const uint32_t num_bottom = get_arg_val<uint32_t>(4);
    const uint32_t start_corner = get_arg_val<uint32_t>(5);
    const uint32_t num_corner = get_arg_val<uint32_t>(6);

    constexpr auto dst_args = TensorAccessorArgs<10>();
    const auto s = TensorAccessor(dst_args, buf_addr, tile_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb_right_mask(cb_right_mask_idx);
    experimental::CircularBuffer cb_bot_mask(cb_bot_mask_idx);
    experimental::CircularBuffer cb_data_out(cb_data_out_idx);

    // ---- Phase 1: generate and push mask tile(s) ----
    using mask_t = MASK_ELEM_UINT;
    constexpr uint32_t TILE = 32;
    if constexpr (has_right_pad) {
        push_right_mask_tile<mask_t, W_mod32, TILE>(cb_right_mask, static_cast<mask_t>(MASK_VALUE));
    }
    if constexpr (has_bottom_pad) {
        push_bottom_mask_tile<mask_t, H_mod32, TILE>(cb_bot_mask, static_cast<mask_t>(MASK_VALUE));
    }

    // ---- Phase 2: write-back loop ----
    // Tiles arrive in the same order as the reader pushes them (right, bottom, corner).

    // Right phase. Maintain (slice, row) incrementally instead of dividing every iteration
    // — RV32IM division is slow. Startup division runs at most once per kernel invocation.
    if constexpr (has_right_pad) {
        uint32_t slice = num_right ? start_right / right_slice_stride : 0u;
        uint32_t row = num_right ? start_right - slice * right_slice_stride : 0u;
        for (uint32_t i = 0; i < num_right; ++i) {
            const uint32_t tile_id = slice * H_tiles * W_tiles + row * W_tiles + (W_tiles - 1u);
            cb_data_out.wait_front(1);
            noc.async_write(cb_data_out, s, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
            noc.async_writes_flushed();
            cb_data_out.pop_front(1);
            ++row;
            if (row == right_slice_stride) {
                row = 0;
                ++slice;
            }
        }
    }

    // Bottom phase. Same incremental pattern as the right phase.
    if constexpr (has_bottom_pad) {
        uint32_t slice = num_bottom ? start_bottom / bottom_slice_stride : 0u;
        uint32_t col = num_bottom ? start_bottom - slice * bottom_slice_stride : 0u;
        for (uint32_t j = 0; j < num_bottom; ++j) {
            const uint32_t tile_id = slice * H_tiles * W_tiles + (H_tiles - 1u) * W_tiles + col;
            cb_data_out.wait_front(1);
            noc.async_write(cb_data_out, s, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
            noc.async_writes_flushed();
            cb_data_out.pop_front(1);
            ++col;
            if (col == bottom_slice_stride) {
                col = 0;
                ++slice;
            }
        }
    }

    // Corner phase
    if constexpr (has_right_pad && has_bottom_pad) {
        for (uint32_t k = 0; k < num_corner; ++k) {
            const uint32_t slice = start_corner + k;
            const uint32_t tile_id = slice * H_tiles * W_tiles + (H_tiles - 1u) * W_tiles + (W_tiles - 1u);
            cb_data_out.wait_front(1);
            noc.async_write(cb_data_out, s, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
            noc.async_writes_flushed();
            cb_data_out.pop_front(1);
        }
    }

    noc.async_write_barrier();
}
