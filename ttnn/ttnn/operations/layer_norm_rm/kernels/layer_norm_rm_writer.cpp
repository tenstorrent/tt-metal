// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// layer_norm_rm writer — runs on BRISC.
//
// Per-core work: for each of `num_strips` strips, write each chunk back to
// DRAM by draining cb_output_rm. write_sticks_after_untilize<TILE>
// owns the cb_wait_front / noc_async_write / cb_pop_front cycle.
//
// Each chunk produces up to 32 sticks of `chunk_bytes` bytes starting at row
// `strip * 32` of the output tensor, offset `c * chunk_bytes` along W.
//
// Refinement 3 — non-tile-aligned shapes:
//   * H non-aligned: when the strip's global index equals last_strip_idx, the
//     helper is called with `total_num_rows = last_strip_rows < 32` instead of
//     32, so the padded rows compute by the compute kernel are dropped on the
//     way out (the helper still pops BLOCK_SIZE tile-pages from the CB to
//     keep the producer/consumer count balanced).
//   * W non-aligned: the LAST chunk's `row_bytes` is `chunk_bytes_last` (the
//     actual valid byte count for the chunk's W coverage) instead of
//     `chunk_bytes`. The helper writes only those bytes; the padded W
//     positions in the untilized CB never reach DRAM.
//
// CT arg layout:
//   [0]  BLOCK_SIZE        (unused; reserved for future per-chunk policy hooks)
//   [1]  NUM_BLOCKS
//   [2]  chunk_bytes
//   [3]  chunk_bytes_last  Refinement 3 — actual valid bytes for the last chunk
//   [4]  last_strip_idx    Refinement 3 — index of the global last strip
//   [5]  last_strip_rows   Refinement 3 — # of valid rows in the global last strip
//   [6..] TensorAccessorArgs(output_tensor)
//
// RT arg layout:
//   [0] output_addr
//   [1] num_strips_for_core
//   [2] start_strip_id

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_output_rm = 16;
}  // namespace

void kernel_main() {
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t chunk_bytes_last = get_compile_time_arg_val(3);
    constexpr uint32_t last_strip_idx = get_compile_time_arg_val(4);
    constexpr uint32_t last_strip_rows = get_compile_time_arg_val(5);
    (void)BLOCK_SIZE;

    constexpr auto output_args = TensorAccessorArgs<6>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_strips = get_arg_val<uint32_t>(1);
    const uint32_t start_strip = get_arg_val<uint32_t>(2);

    const auto output_accessor = TensorAccessor(output_args, output_addr);

    for (uint32_t i = 0; i < num_strips; ++i) {
        const uint32_t strip = start_strip + i;
        const uint32_t strip_start_row = strip * 32;

        // Refinement 3 — H non-aligned: pass the partial row count for the
        // global last strip. For tile-aligned shapes last_strip_rows = 32 and
        // last_strip_idx is the natural global last strip — both paths produce
        // identical writes.
        const uint32_t rows_this_strip = (strip == last_strip_idx) ? last_strip_rows : 32u;

        for (uint32_t c = 0; c < NUM_BLOCKS; ++c) {
            // Refinement 3 — W non-aligned: the LAST chunk's byte count is the
            // remaining valid bytes (W*bpe - prev_chunk_offsets), not a full
            // chunk. The helper writes only those bytes per stick.
            const uint32_t chunk_bytes_this = (c == NUM_BLOCKS - 1) ? chunk_bytes_last : chunk_bytes;
            dataflow_kernel_lib::write_sticks_after_untilize<cb_output_rm>(
                output_accessor,
                /*total_num_rows=*/rows_this_strip,
                /*row_bytes=*/chunk_bytes_this,
                /*start_page=*/strip_start_row,
                /*byte_offset_within_page=*/c * chunk_bytes);
        }
    }
}
