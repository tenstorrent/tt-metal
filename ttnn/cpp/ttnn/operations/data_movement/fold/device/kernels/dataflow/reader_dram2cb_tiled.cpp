// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    constexpr uint32_t tiles_per_channel_dim = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_width_dim = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(2);

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_block_id = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);

    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);

    // Initialize interleaved address generator for DRAM access
    constexpr auto src_args = TensorAccessorArgs<3>();
    const auto s = TensorAccessor(src_args, src_addr, tile_bytes);

    // Process each block of data
    uint32_t end_block_id = start_block_id + num_blocks;
    for (uint32_t i = start_block_id; i < end_block_id; ++i) {
        // Reserve space in the circular buffer for a row of tiles
        for (uint32_t j = 0; j < tiles_per_width_dim; ++j) {
            cb_reserve_back(cb_id_in0, tiles_per_channel_dim);
            uint64_t l1_write_addr = get_write_ptr(cb_id_in0);

            // Read each tile in the current row
            for (uint32_t k = 0; k < tiles_per_channel_dim; ++k) {
                // Calculate tile index and read from DRAM to L1
                uint32_t tile_index = tiles_per_width_dim * tiles_per_channel_dim * i + tiles_per_channel_dim * j + k;
                noc_async_read_tile(tile_index, s, l1_write_addr);

                l1_write_addr += tile_bytes;
            }

            noc_async_read_barrier();

            // Ensure all async reads are complete before proceeding
            // Push the completed row of tiles to the circular buffer
            cb_push_back(cb_id_in0, tiles_per_channel_dim);
        }
    }
}
