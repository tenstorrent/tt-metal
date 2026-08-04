// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer for rms_norm (BRISC, NoC1).
//
// Mirror image of the reader — same (row-block, width-chunk) loop nest, same
// WT_CHUNK transaction granularity, so both NoC halves are batched the same
// way (a reader-only batching lever just moves the bottleneck across the CB):
//   * TILE build      : cb_output_tiles  -> whole output tiles
//   * ROW_MAJOR build  : cb_output_sticks -> output sticks, W*elem bytes each
//
// The ROW_MAJOR path uses dataflow_kernel_lib::write_sticks_after_untilize,
// which is exactly the consumer contract of
// compute_kernel_lib::untilize<WT_CHUNK>(rows): it waits WT_CHUNK tile-sized
// pages per tile-row, writes only the VALID sticks (so trailing rows of a short
// final tile-row are never written) and only `row_bytes` of each (so the W tile
// padding is never written).
//
// Pass B runs once per row-block regardless of regime, so unlike the reader
// there is no pass loop here.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_output_tiles = 8;
constexpr uint32_t cb_output_sticks = 9;
constexpr uint32_t TILE_DIM = 32;
}  // namespace

void kernel_main() {
    // ---- compile-time knobs (all from rms_norm_program_descriptor.py) -----
    constexpr uint32_t IS_TILE = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_W_CHUNKS = get_compile_time_arg_val(3);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(4);
    constexpr uint32_t ELEM_BYTES = get_compile_time_arg_val(5);
    constexpr uint32_t R_RM = get_compile_time_arg_val(6);
    constexpr uint32_t W_ELEMS = get_compile_time_arg_val(7);
    constexpr auto out_args = TensorAccessorArgs<8>();

    constexpr bool RM = (IS_TILE == 0);
    constexpr uint32_t CHUNK_ROW_BYTES = WT_CHUNK * TILE_DIM * ELEM_BYTES;
    constexpr uint32_t LAST_CHUNK_ROW_BYTES = W_ELEMS * ELEM_BYTES - (NUM_W_CHUNKS - 1) * CHUNK_ROW_BYTES;

    // ---- runtime work assignment -----------------------------------------
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);  // this core's first tile-row
    const uint32_t num_rows = get_arg_val<uint32_t>(2);   // tile-rows owned by this core

    const auto out_acc = TensorAccessor(out_args, out_addr);
    const uint32_t out_tile_bytes = get_tile_size(cb_output_tiles);

    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
        const uint32_t first_tile_row = row_start + r0;

        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            if constexpr (RM) {
                const uint32_t stick_start = first_tile_row * TILE_DIM;
                uint32_t sticks = rows * TILE_DIM;
                if (stick_start + sticks > R_RM) {
                    sticks = R_RM - stick_start;  // short final tile-row
                }
                const uint32_t row_bytes = (c + 1 == NUM_W_CHUNKS) ? LAST_CHUNK_ROW_BYTES : CHUNK_ROW_BYTES;
                dataflow_kernel_lib::write_sticks_after_untilize<cb_output_sticks>(
                    out_acc, sticks, row_bytes, stick_start, /*byte_offset_within_page=*/c * CHUNK_ROW_BYTES);
            } else {
                for (uint32_t r = 0; r < rows; ++r) {
                    const uint32_t tile_base = (first_tile_row + r) * WT + c * WT_CHUNK;
                    cb_wait_front(cb_output_tiles, WT_CHUNK);
                    uint32_t l1_addr = get_read_ptr(cb_output_tiles);
                    for (uint32_t w = 0; w < WT_CHUNK; ++w) {
                        noc_async_write_tile(tile_base + w, out_acc, l1_addr);
                        l1_addr += out_tile_bytes;
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_output_tiles, WT_CHUNK);
                }
            }
        }
    }
}
