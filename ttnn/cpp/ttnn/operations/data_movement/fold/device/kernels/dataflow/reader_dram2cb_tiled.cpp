// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    constexpr uint32_t tiles_per_channel_dim = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_width_dim = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(2);

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_block_id = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);

    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);

    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::Noc noc;

    // Initialize interleaved address generator for DRAM access
    constexpr auto src_args = TensorAccessorArgs<3>();
    const auto s = TensorAccessor(src_args, src_addr);

    // Process each block of data
    uint32_t end_block_id = start_block_id + num_blocks;
    for (uint32_t i = start_block_id; i < end_block_id; ++i) {
        // Reserve space in the circular buffer for a row of tiles
        for (uint32_t j = 0; j < tiles_per_width_dim; ++j) {
            cb_in0.reserve_back(tiles_per_channel_dim);

            // Read each tile in the current row
            for (uint32_t k = 0; k < tiles_per_channel_dim; ++k) {
                // Calculate tile index and read from DRAM to L1
                uint32_t tile_index = tiles_per_width_dim * tiles_per_channel_dim * i + tiles_per_channel_dim * j + k;
                noc.async_read<experimental::Noc::TxnIdMode::DISABLED, tile_bytes>(
                    s, cb_in0, tile_bytes, {.page_id = tile_index}, {.offset_bytes = k * tile_bytes});
            }

            noc.async_read_barrier();

            // Ensure all async reads are complete before proceeding
            // Push the completed row of tiles to the circular buffer
            cb_in0.push_back(tiles_per_channel_dim);
        }
    }
}
