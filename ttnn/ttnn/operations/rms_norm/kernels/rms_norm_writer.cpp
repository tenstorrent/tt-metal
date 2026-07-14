// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for rms_norm (dataflow).
//
// TILE regime: the compute pushes normalized tiles in the exact global page
// order, so the writer drains cb_output_tiles into a contiguous tile-id range.
// ROW_MAJOR regime: the compute untilizes each W-block; the writer streams the
// valid sticks back per (tile-row, W-block), matching the reader's stick
// geometry (only the true rows/cols of each partial block are written).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_output_rm = 17;
constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;
}  // namespace

void kernel_main() {
    constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(0);
    constexpr uint32_t origin_W = get_compile_time_arg_val(1);
    constexpr uint32_t origin_H = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_image = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t W_BLOCK_TILES = get_compile_time_arg_val(5);
    constexpr uint32_t num_w_blocks = get_compile_time_arg_val(6);
    constexpr uint32_t output_elt = get_compile_time_arg_val(7);
    constexpr auto dst_args = TensorAccessorArgs<8>();

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_tile_row = get_arg_val<uint32_t>(1);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(2);

    constexpr uint32_t wblock_cols = W_BLOCK_TILES * TILE_W;

    if constexpr (IS_ROW_MAJOR) {
        const auto acc = TensorAccessor(dst_args, dst_addr);
        for (uint32_t t = 0; t < num_tile_rows; ++t) {
            uint32_t tr = start_tile_row + t;
            uint32_t image = tr / tiles_per_image;
            uint32_t local_tr = tr - image * tiles_per_image;
            uint32_t base_stick = image * origin_H + local_tr * TILE_H;
            uint32_t num_rows = origin_H - local_tr * TILE_H;
            if (num_rows > TILE_H) {
                num_rows = TILE_H;
            }
            for (uint32_t b = 0; b < num_w_blocks; ++b) {
                uint32_t cols = origin_W - b * wblock_cols;
                if (cols > wblock_cols) {
                    cols = wblock_cols;
                }
                dataflow_kernel_lib::write_sticks_after_untilize<cb_output_rm>(
                    acc, num_rows, cols * output_elt, base_stick, b * wblock_cols * output_elt);
            }
        }
    } else {
        // TILE regime. The compute pushes W_BLOCK_TILES output tiles per W-block
        // (in global page order). DOUBLE_BUFFER (Refinement 3): drain a whole
        // block with ONE barrier instead of write-one/barrier-per-tile — the
        // per-tile write+barrier was ~24% of device time (not hidden by compute;
        // it back-pressures the pass-2 mul through cb_output_tiles). cb_output_tiles
        // is 2*W_BLOCK_TILES deep, and both compute-push and writer-pop granularity
        // are W_BLOCK_TILES, so the W_BLOCK_TILES-run at the CB front never wraps.
        uint32_t tile_bytes = get_tile_size(cb_output_tiles);
        const auto acc = TensorAccessor(dst_args, dst_addr, tile_bytes);
        for (uint32_t t = 0; t < num_tile_rows; ++t) {
            uint32_t tr = start_tile_row + t;
            for (uint32_t b = 0; b < num_w_blocks; ++b) {
                cb_wait_front(cb_output_tiles, W_BLOCK_TILES);
                uint32_t rp = get_read_ptr(cb_output_tiles);
                uint32_t base_tile = tr * Wt + b * W_BLOCK_TILES;
                for (uint32_t wt = 0; wt < W_BLOCK_TILES; ++wt) {
                    noc_async_write_page(base_tile + wt, acc, rp + wt * tile_bytes);
                }
                noc_async_write_barrier();
                cb_pop_front(cb_output_tiles, W_BLOCK_TILES);
            }
        }
    }
}
