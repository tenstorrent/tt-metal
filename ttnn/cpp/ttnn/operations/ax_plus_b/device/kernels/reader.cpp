// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

//
// Reader for elemwise y = ax + b
// Assumptions:
// - a, x and b are of the same tile size.
//
void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t x_addr = get_arg_val<uint32_t>(1);
    uint32_t b_addr = get_arg_val<uint32_t>(2);

    // shared between all 3 tensors
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_x = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_b = tt::CBIndex::c_2;

    // 3 CB ids => 3 TensorAccessorArgs blocks
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto x_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    uint32_t src_tile_bytes = get_tile_size(cb_id_a);

    const auto a_accessor = TensorAccessor(a_args, a_addr, src_tile_bytes);
    const auto x_accessor = TensorAccessor(x_args, x_addr, src_tile_bytes);
    const auto b_accessor = TensorAccessor(b_args, b_addr, src_tile_bytes);

    auto read_tile_for_buffer = [](uint32_t tile_index, const auto& accessor, uint32_t cb_id) {
        cb_reserve_back(cb_id, 1);
        noc_async_read_tile(tile_index, accessor, get_write_ptr(cb_id));
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    };

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        read_tile_for_buffer(i, b_accessor, cb_id_b);
        read_tile_for_buffer(i, a_accessor, cb_id_a);
        read_tile_for_buffer(i, x_accessor, cb_id_x);
    }
}
