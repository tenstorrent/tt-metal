// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t tiles_per_channel_dim = get_arg(args::tiles_per_channel_dim);
    constexpr uint32_t tiles_per_width_dim = get_arg(args::tiles_per_width_dim);

    auto start_block_id = get_arg(args::start_block_id);
    auto num_blocks = get_arg(args::num_blocks);

    DataflowBuffer cb_in0(dfb::src0);
    const uint32_t tile_bytes = cb_in0.get_entry_size();

    // Interleaved DRAM access via the bound input tensor.
    const auto s = TensorAccessor(tensor::src);

    Noc noc;

    // Process each block of data
    uint32_t end_block_id = start_block_id + num_blocks;
    for (uint32_t i = start_block_id; i < end_block_id; ++i) {
        // Reserve space in the circular buffer for a row of tiles
        for (uint32_t j = 0; j < tiles_per_width_dim; ++j) {
            cb_in0.reserve_back(tiles_per_channel_dim);
            uint32_t l1_offset = 0;

            // Read each tile in the current row
            for (uint32_t k = 0; k < tiles_per_channel_dim; ++k) {
                // Calculate tile index and read from DRAM to L1
                uint32_t tile_index = tiles_per_width_dim * tiles_per_channel_dim * i + tiles_per_channel_dim * j + k;
                noc.async_read(s, cb_in0, tile_bytes, {.page_id = tile_index}, {.offset_bytes = l1_offset});
                l1_offset += tile_bytes;
            }

            noc.async_read_barrier();

            // Ensure all async reads are complete before proceeding
            // Push the completed row of tiles to the circular buffer
            cb_in0.push_back(tiles_per_channel_dim);
        }
    }
}
