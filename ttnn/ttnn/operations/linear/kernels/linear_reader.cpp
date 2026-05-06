// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// linear reader (Phase 0): single core, NCRISC.
//
// Pushes (in order):
//   - bias  → cb_bias_tiles    (Nt tiles, only when has_bias=1)
//   - input → cb_input_tiles   (Mt*Kt tiles, tile-row-major)
//   - weight→ cb_weight_tiles  (Kt*Nt tiles, tile-row-major)
//
// Bias-first ordering follows the toy_binary_in_place pattern: the matmul
// helper waits on the full Mt*Kt input block before starting, but the bias
// helper waits on cb_bias_tiles only after matmul completes. We still push
// bias first so it's ready by the time the matmul output partials drain into
// add_bias_bcast_rows — it's the cheap CB and is sized to all Nt tiles, so
// pushing it first never starves the input/weight read path. Per-tile push
// (rather than bulk reserve+read+push) keeps NoC reads streaming while
// compute is reading earlier tiles, even though Phase 0's CBs are big enough
// for the bulk pattern.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ── Compile-time args ───────────────────────────────────────────────
    constexpr uint32_t has_bias = get_compile_time_arg_val(0);
    constexpr uint32_t input_num_tiles = get_compile_time_arg_val(1);   // Mt*Kt
    constexpr uint32_t weight_num_tiles = get_compile_time_arg_val(2);  // Kt*Nt
    constexpr uint32_t bias_num_tiles = get_compile_time_arg_val(3);    // Nt or 0

    // TensorAccessorArgs are at the END of the CT arg list, chained via
    // next_compile_time_args_offset(). Bias is declared unconditionally with
    // a no-arg placeholder when absent so the offset chain stays valid.
    constexpr auto input_args = TensorAccessorArgs<4>();
    constexpr auto weight_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();

    // ── Runtime args ────────────────────────────────────────────────────
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t weight_addr = get_arg_val<uint32_t>(1);
    [[maybe_unused]] const uint32_t bias_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_input_tiles = 0;
    constexpr uint32_t cb_weight_tiles = 1;
    constexpr uint32_t cb_bias_tiles = 2;

    // ── Bias (if present) — Nt tiles, page i is tile (0, i) of bias DRAM ─
    if constexpr (has_bias) {
        const uint32_t bias_tile_bytes = get_tile_size(cb_bias_tiles);
        const auto accessor = TensorAccessor(bias_args, bias_addr, bias_tile_bytes);
        for (uint32_t i = 0; i < bias_num_tiles; ++i) {
            cb_reserve_back(cb_bias_tiles, 1);
            const uint32_t l1_write_addr = get_write_ptr(cb_bias_tiles);
            noc_async_read_tile(i, accessor, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_bias_tiles, 1);
        }
    }

    // ── Input — Mt*Kt tiles, tile-row-major (page i = tile linear index) ─
    {
        const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
        const auto accessor = TensorAccessor(input_args, input_addr, tile_bytes);
        for (uint32_t i = 0; i < input_num_tiles; ++i) {
            cb_reserve_back(cb_input_tiles, 1);
            const uint32_t l1_write_addr = get_write_ptr(cb_input_tiles);
            noc_async_read_tile(i, accessor, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_input_tiles, 1);
        }
    }

    // ── Weight — Kt*Nt tiles, tile-row-major ─────────────────────────────
    {
        const uint32_t tile_bytes = get_tile_size(cb_weight_tiles);
        const auto accessor = TensorAccessor(weight_args, weight_addr, tile_bytes);
        for (uint32_t i = 0; i < weight_num_tiles; ++i) {
            cb_reserve_back(cb_weight_tiles, 1);
            const uint32_t l1_write_addr = get_write_ptr(cb_weight_tiles);
            noc_async_read_tile(i, accessor, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_weight_tiles, 1);
        }
    }
}
