// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for toy_binary_in_place.
// Loads B tiles first (fewer or equal count), then A tiles.
// B-first ordering prevents deadlocks: compute needs B alongside A,
// so B must be available before A streaming begins.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t a_start_id = get_arg_val<uint32_t>(1);
    uint32_t b_addr = get_arg_val<uint32_t>(2);
    uint32_t b_start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t a_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t b_num_tiles = get_compile_time_arg_val(1);
    constexpr auto a_args = TensorAccessorArgs<2>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;

    // Read B tiles first — compute needs B available alongside A
    {
        uint32_t tile_bytes = get_tile_size(cb_b);
        const auto accessor = TensorAccessor(b_args, b_addr, tile_bytes);
        for (uint32_t i = b_start_id; i < b_start_id + b_num_tiles; i++) {
            cb_reserve_back(cb_b, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_b);
            noc_async_read_tile(i, accessor, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_b, 1);
        }
    }

    // Read A tiles
    {
        uint32_t tile_bytes = get_tile_size(cb_a);
        const auto accessor = TensorAccessor(a_args, a_addr, tile_bytes);
        for (uint32_t i = a_start_id; i < a_start_id + a_num_tiles; i++) {
            cb_reserve_back(cb_a, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_a);
            noc_async_read_tile(i, accessor, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_a, 1);
        }
    }
}
