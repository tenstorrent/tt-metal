// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for iterative_topk with TILE layout input.
// Reads width_tiles tile pages per height-tile batch from DRAM into the input CB.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_height_tile = get_arg_val<uint32_t>(1);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t input_tile_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t width_tiles = get_compile_time_arg_val(2);

    constexpr uint32_t accessor_offset = 3;
    constexpr auto input_args = TensorAccessorArgs<accessor_offset>();
    const auto input_accessor = TensorAccessor(input_args, input_addr);

    for (uint32_t ht = start_height_tile; ht < end_height_tile; ht++) {
        uint32_t base_page = ht * width_tiles;
        for (uint32_t wt = 0; wt < width_tiles; wt++) {
            cb_reserve_back(cb_input, 1);
            noc_async_read_page(base_page + wt, input_accessor, get_write_ptr(cb_input));
            noc_async_read_barrier();
            cb_push_back(cb_input, 1);
        }
    }
}
