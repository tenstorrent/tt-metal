// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// glu_fused — Reader kernel.
//
// For each output tile (out_idx in [start_out_tile_id, start + num_per_core)):
//   row_idx     = out_idx / Wt_half                 (tile-row index in the input)
//   col_in_half = out_idx % Wt_half                 (tile-col within either half)
//   a_tile_idx  = row_idx * Wt + col_in_half        (first half of input)
//   b_tile_idx  = row_idx * Wt + Wt_half + col_in_half  (second half of input)
//   where Wt = 2 * Wt_half.
//
// Pushes one A tile to cb_input_a and one B tile to cb_input_b per output tile,
// alternating CBs each iteration. Both CBs are double-buffered so the reader
// can stage tile (i+1) while compute consumes tile i.
//
// CT args: [cb_input_a, cb_input_b, Wt_half, TensorAccessorArgs(input)...]
// RT args: [src_addr, num_output_tiles, start_out_tile_id]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_out_tile_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_input_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input_b = get_compile_time_arg_val(1);
    constexpr uint32_t Wt_half = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = 2 * Wt_half;
    constexpr auto src_args = TensorAccessorArgs<3>();

    const uint32_t tile_bytes = get_tile_size(cb_input_a);
    const auto src_accessor = TensorAccessor(src_args, src_addr, tile_bytes);

    const uint32_t end_out_tile_id = start_out_tile_id + num_output_tiles;
    for (uint32_t out_idx = start_out_tile_id; out_idx < end_out_tile_id; ++out_idx) {
        const uint32_t row_idx = out_idx / Wt_half;
        const uint32_t col_in_half = out_idx % Wt_half;
        const uint32_t a_tile_idx = row_idx * Wt + col_in_half;
        const uint32_t b_tile_idx = a_tile_idx + Wt_half;

        // Push A tile from first half.
        cb_reserve_back(cb_input_a, 1);
        const uint32_t l1_write_addr_a = get_write_ptr(cb_input_a);
        noc_async_read_tile(a_tile_idx, src_accessor, l1_write_addr_a);
        noc_async_read_barrier();
        cb_push_back(cb_input_a, 1);

        // Push B tile from second half.
        cb_reserve_back(cb_input_b, 1);
        const uint32_t l1_write_addr_b = get_write_ptr(cb_input_b);
        noc_async_read_tile(b_tile_idx, src_accessor, l1_write_addr_b);
        noc_async_read_barrier();
        cb_push_back(cb_input_b, 1);
    }
}
