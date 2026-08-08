// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Drain-only writer for fused distributed GroupNorm (ring_size==1 / is_local).
 * Populates epsilon CB, then drains output_cb to DRAM.
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(1);
    constexpr uint32_t block_size = get_compile_time_arg_val(2);
    constexpr uint32_t epsilon_cb = get_compile_time_arg_val(3);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(4);
    constexpr auto output_args = TensorAccessorArgs<5>();

    size_t arg_idx = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t output_tile_bytes = get_tile_size(output_cb);
    const auto output_accessor = TensorAccessor(output_args, output_addr);

    cb_reserve_back(epsilon_cb, 1);
    volatile tt_l1_ptr uint32_t* eps_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(epsilon_cb));
    eps_ptr[0] = eps_bits;
    cb_push_back(epsilon_cb, 1);

    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            const uint32_t tiles_in_block =
                ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
            cb_wait_front(output_cb, block_size);
            uint32_t rd = get_read_ptr(output_cb);
            for (uint32_t i = 0; i < tiles_in_block; i++) {
                const uint32_t out_idx = tile_row * num_tile_cols + col_tile + i;
                noc_async_write_tile(out_idx, output_accessor, rd);
                rd += output_tile_bytes;
            }
            noc_async_writes_flushed();
            cb_pop_front(output_cb, block_size);
        }
    }
    noc_async_write_barrier();
}
