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
    constexpr uint32_t stride_height = get_arg(args::stride_height);

    const uint32_t start_super_block_id = get_arg(args::start_block_id);
    const uint32_t num_super_blocks = get_arg(args::num_blocks);

    DataflowBuffer cb_in0(dfb::src0);
    const uint32_t tile_bytes = cb_in0.get_entry_size();

    const auto s = TensorAccessor(tensor::src);
    Noc noc;

    // Each super-block = `stride_height` consecutive input rows; the writer gathers them into scratch before
    // emitting one aligned output row, so work must split at super-block granularity across cores.
    const uint32_t end_super_block_id = start_super_block_id + num_super_blocks;
    for (uint32_t sb = start_super_block_id; sb < end_super_block_id; ++sb) {
        const uint32_t input_h_base = sb * stride_height;
        for (uint32_t local_h = 0; local_h < stride_height; ++local_h) {
            const uint32_t input_h = input_h_base + local_h;
            for (uint32_t w_tile = 0; w_tile < tiles_per_width_dim; ++w_tile) {
                cb_in0.reserve_back(tiles_per_channel_dim);
                uint32_t l1_offset = 0;
                for (uint32_t c_tile = 0; c_tile < tiles_per_channel_dim; ++c_tile) {
                    const uint32_t tile_index =
                        input_h * tiles_per_width_dim * tiles_per_channel_dim + w_tile * tiles_per_channel_dim + c_tile;
                    noc.async_read(s, cb_in0, tile_bytes, {.page_id = tile_index}, {.offset_bytes = l1_offset});
                    l1_offset += tile_bytes;
                }
                noc.async_read_barrier();
                cb_in0.push_back(tiles_per_channel_dim);
            }
        }
    }
}
