// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for the deepseek_prefill::dummy_op op.
// For num_iter iterations, reads this core's assigned chunk of tiles from
// input_tensor (DRAM-interleaved) into the reader→writer circular buffer.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t READ_BATCH = 8;  // tiles per NOC barrier; must be <= CB depth.

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t total_tiles = get_arg_val<uint32_t>(1);
    const uint32_t core_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_tile = get_compile_time_arg_val(0);
    constexpr uint32_t num_iter = get_compile_time_arg_val(1);
    constexpr uint32_t num_cores = get_compile_time_arg_val(2);

    constexpr auto input_args = TensorAccessorArgs<3>();
    const auto input_accessor = TensorAccessor(input_args, input_addr, get_tile_size(cb_tile));

    // Split total_tiles across num_cores cores. Each core's range is
    //   [ (N * core_id)     / num_cores,
    //     (N * (core_id+1)) / num_cores )
    // distributing the remainder across the tail cores.
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
            const uint32_t batch = remaining < READ_BATCH ? remaining : READ_BATCH;

            cb_reserve_back(cb_tile, batch);
            uint32_t l1_write_addr = get_write_ptr(cb_tile);
            for (uint32_t i = 0; i < batch; ++i) {
                noc_async_read_tile(tile_idx + i, input_accessor, l1_write_addr);
                l1_write_addr += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_tile, batch);

            tile_idx += batch;
        }
    }
}
