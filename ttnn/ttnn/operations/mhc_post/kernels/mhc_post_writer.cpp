// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post writer (BRISC / NoC1) — plain-2d regime.
//
// store_block(row, b): the block's output window is n * block_c pages laid out [j][c]
// (compute pushes per tile; a ragged tail is padded to the nominal count by compute).
// Each valid tile is written as soon as it is pushed (cumulative wait on the window),
// one write barrier per block, then the whole nominal window is popped.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(0);
constexpr uint32_t n = get_compile_time_arg_val(1);
constexpr uint32_t block_c = get_compile_time_arg_val(2);
constexpr uint32_t tensor_c_tiles = get_compile_time_arg_val(3);
constexpr auto out_args = TensorAccessorArgs<4>();

}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
    const uint32_t c_start = get_arg_val<uint32_t>(3);
    const uint32_t num_c = get_arg_val<uint32_t>(4);
    // args 5, 6: (g_r, g_c) — consumed by the fused regimes only.

    const uint32_t tile_bytes = get_tile_size(cb_out_tiles);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);
    const uint32_t num_blocks = (num_c + block_c - 1) / block_c;
    constexpr uint32_t window = n * block_c;

    for (uint32_t r = 0; r < num_rows; ++r) {
        const uint32_t row = row_start + r;
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t c0 = c_start + b * block_c;
            const uint32_t extent = (num_c - b * block_c) < block_c ? (num_c - b * block_c) : block_c;
            uint32_t k = 0;  // position in the compute-push order (j-major, c-minor)
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t page0 = (row * n + j) * tensor_c_tiles + c0;
                for (uint32_t c = 0; c < extent; ++c, ++k) {
                    cb_wait_front(cb_out_tiles, k + 1);
                    noc_async_write_tile(page0 + c, out_acc, get_read_ptr(cb_out_tiles) + k * tile_bytes);
                }
            }
            cb_wait_front(cb_out_tiles, window);  // padding pages of a ragged tail
            noc_async_write_barrier();
            cb_pop_front(cb_out_tiles, window);
        }
    }
}
