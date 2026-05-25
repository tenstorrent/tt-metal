// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Softmax writer — Refinement 1 (chunked, per-tile streaming).
//
// Per-core work: for each of the `num_strips` strips, wait for tiles to arrive
// in cb_output_tiles ONE AT A TIME and write them back to DRAM at the tile ids
// that mirror the reader's input pattern (output shape == input shape).
//
// Per-tile streaming (vs strip-at-a-time) is the partner-side change to the
// reader's per-tile streaming: cb_output_tiles is sized at 2 pages
// (double-buffered) so it does not scale with `reduce_dim_tiles`.
//
// Strip-to-tile mapping mirrors the reader; see softmax_reader.cpp.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t cb_output_tiles = 16;
}  // namespace

void kernel_main() {
    constexpr uint32_t dim_is_row = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t reduce_dim_tiles = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_strips = get_arg_val<uint32_t>(1);
    const uint32_t start_strip = get_arg_val<uint32_t>(2);

    const uint32_t tile_bytes = get_tile_size(cb_output_tiles);
    const auto accessor = TensorAccessor(dst_args, dst_addr, tile_bytes);

    for (uint32_t i = 0; i < num_strips; ++i) {
        const uint32_t s = start_strip + i;

        uint32_t base_tile_id;
        uint32_t stride;
        if constexpr (dim_is_row != 0) {
            base_tile_id = s * Wt;
            stride = 1;
        } else {
            const uint32_t nc = s / Wt;
            const uint32_t wt = s - nc * Wt;
            base_tile_id = nc * (Ht * Wt) + wt;
            stride = Wt;
        }

        for (uint32_t t = 0; t < reduce_dim_tiles; ++t) {
            const uint32_t tile_id = base_tile_id + t * stride;
            cb_wait_front(cb_output_tiles, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
            noc_async_write_tile(tile_id, accessor, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_output_tiles, 1);
        }
    }
}
