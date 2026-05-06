// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// linear writer (Phase 0): single core, BRISC.
//
// Drains cb_output_tiles to DRAM. The matmul (and bias) helpers pack tiles
// in subblock-major order with out_subblock_h = out_subblock_w = 1, which
// is identical to tile-row-major (m, n) order: the writer reads page
// m*Nt + n directly off the natural DRAM-interleaved tile-linear index.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ── Compile-time args ───────────────────────────────────────────────
    constexpr uint32_t output_num_tiles = get_compile_time_arg_val(0);  // Mt*Nt
    constexpr auto output_args = TensorAccessorArgs<1>();

    // ── Runtime args ────────────────────────────────────────────────────
    const uint32_t output_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_output_tiles = 16;
    const uint32_t tile_bytes = get_tile_size(cb_output_tiles);
    const auto accessor = TensorAccessor(output_args, output_addr, tile_bytes);

    for (uint32_t i = 0; i < output_num_tiles; ++i) {
        cb_wait_front(cb_output_tiles, 1);
        const uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
        noc_async_write_tile(i, accessor, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_output_tiles, 1);
    }
}
