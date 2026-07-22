// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for rms_norm.
//
// Row-parallel: this core writes the output tile-rows [row_start, row_start+num_rows).
//
//   TILE : write whole output tiles from cb_out (tile_id = r*Wt + b*BS + wt).
//   RM   : write untilized row-major sticks from cb_out_sticks via the tilize
//          dataflow helper. It writes exactly `row_bytes` valid columns per
//          stick and only the valid rows of the last (partial-H) tile-row, so
//          non-tile-aligned W/H land correctly with no host-side slice.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_out_sticks = 17;
constexpr uint32_t TILE_DIM = 32;
}  // namespace

void kernel_main() {
    constexpr uint32_t Ht_img = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(3);
    constexpr uint32_t origin_H = get_compile_time_arg_val(4);
    constexpr uint32_t origin_W = get_compile_time_arg_val(5);
    constexpr bool IS_RM = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t out_elem = get_compile_time_arg_val(7);
    constexpr uint32_t out_page = get_compile_time_arg_val(8);

    constexpr auto out_args = TensorAccessorArgs<9>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);

    const auto out_accessor = TensorAccessor(out_args, dst_addr, out_page);

    for (uint32_t i = 0; i < num_rows; ++i) {
        const uint32_t r = row_start + i;
        const uint32_t image = r / Ht_img;
        const uint32_t ht_in_img = r % Ht_img;
        const uint32_t base_stick = image * origin_H + ht_in_img * TILE_DIM;  // RM
        uint32_t valid_rows = origin_H - ht_in_img * TILE_DIM;                // RM
        if (valid_rows > TILE_DIM) {
            valid_rows = TILE_DIM;
        }
        const uint32_t row_tile_base = r * Wt;  // TILE

        for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
            if constexpr (IS_RM) {
                const uint32_t col0 = b * BLOCK_SIZE * TILE_DIM;
                uint32_t cols = origin_W - col0;
                if (cols > BLOCK_SIZE * TILE_DIM) {
                    cols = BLOCK_SIZE * TILE_DIM;
                }
                dataflow_kernel_lib::write_sticks_after_untilize<cb_out_sticks>(
                    out_accessor, valid_rows, cols * out_elem, base_stick, col0 * out_elem);
            } else {
                // Coalesce the whole block behind ONE barrier (writer twin of the
                // reader's batched reads).
                const uint32_t tile_bytes = get_tile_size(cb_out);
                cb_wait_front(cb_out, BLOCK_SIZE);
                uint32_t l1 = get_read_ptr(cb_out);
                for (uint32_t wt = 0; wt < BLOCK_SIZE; ++wt) {
                    noc_async_write_tile(row_tile_base + b * BLOCK_SIZE + wt, out_accessor, l1);
                    l1 += tile_bytes;
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, BLOCK_SIZE);
            }
        }
    }
}
