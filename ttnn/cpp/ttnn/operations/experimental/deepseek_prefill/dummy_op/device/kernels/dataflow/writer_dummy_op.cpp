// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for the deepseek_prefill::dummy_op op.
// For num_iter iterations, drains this core's assigned chunk of tiles from
// the reader→writer CB and writes them back to input_tensor at the same DRAM
// addresses (in-place).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t WRITE_BATCH = 8;  // tiles per NOC barrier; must be <= CB depth.

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t total_tiles = get_arg_val<uint32_t>(1);
    const uint32_t core_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_tile = get_compile_time_arg_val(0);
    constexpr uint32_t num_iter = get_compile_time_arg_val(1);
    constexpr uint32_t num_cores = get_compile_time_arg_val(2);

    constexpr auto output_args = TensorAccessorArgs<3>();
    const auto output_accessor = TensorAccessor(output_args, output_addr, get_tile_size(cb_tile));

    // Matches the reader's split: each core writes back exactly the tiles it
    // read.
    const uint32_t my_start = (total_tiles * core_id) / num_cores;
    const uint32_t my_end = (total_tiles * (core_id + 1)) / num_cores;
    const uint32_t my_num_tiles = my_end - my_start;

    if (my_num_tiles == 0) {
        return;
    }

    const uint32_t tile_bytes = get_tile_size(cb_tile);

    for (uint32_t iter = 0; iter < num_iter; ++iter) {
        uint32_t tile_idx = my_start;
        while (tile_idx < my_end) {
            const uint32_t remaining = my_end - tile_idx;
            const uint32_t batch = remaining < WRITE_BATCH ? remaining : WRITE_BATCH;

            cb_wait_front(cb_tile, batch);
            uint32_t l1_read_addr = get_read_ptr(cb_tile);
            for (uint32_t i = 0; i < batch; ++i) {
                noc_async_write_tile(tile_idx + i, output_accessor, l1_read_addr);
                l1_read_addr += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_tile, batch);

            tile_idx += batch;
        }
    }
}
