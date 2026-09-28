// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "tt-metalium/constants.hpp"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

using namespace tt::data_movement::common;

void kernel_main() {
    constexpr uint32_t input_width = get_arg(args::input_width);
    constexpr uint32_t stride_height = get_arg(args::stride_height);
    constexpr uint32_t stride_width = get_arg(args::stride_width);
    // `c_bytes` = logical C * elem_size; `c_padded_bytes` = untilize row stride (C_tiles * TILE_WIDTH * elem_size).
    constexpr uint32_t c_bytes = get_arg(args::c_bytes);
    constexpr uint32_t c_padded_bytes = get_arg(args::c_padded_bytes);
    constexpr uint32_t tiles_per_channel_dim = get_arg(args::tiles_per_channel_dim);
    constexpr uint32_t tiles_per_width_dim = get_arg(args::tiles_per_width_dim);

    const uint32_t start_block_id = get_arg(args::start_block_id);
    const uint32_t num_blocks = get_arg(args::num_blocks);
    uint32_t patch_height_offset = get_arg(args::patch_height_offset);
    uint32_t curr_out_page = get_arg(args::output_offset);

    constexpr uint32_t output_width = input_width / stride_width;

    const auto dst = TensorAccessor(tensor::dst);
    Noc noc;
    DataflowBuffer dfb_in1(dfb::in1);

    const uint32_t end_block_id = start_block_id + num_blocks;
    for (uint32_t block_id = start_block_id; block_id < end_block_id; block_id++) {
        uint32_t remaining_width = input_width;
        uint32_t out_page = curr_out_page;
        uint32_t stride_w_idx = 0;
        // Per-input-row patch base within the output stick: `(h % sh) * sw` slots.
        const uint32_t row_patch_base = patch_height_offset * stride_width;

        for (uint32_t tile_idx = 0; tile_idx < tiles_per_width_dim; tile_idx++) {
            dfb_in1.wait_front(tiles_per_channel_dim);
            const uint32_t src_base = dfb_in1.get_read_ptr();

            const uint32_t width_limit =
                (remaining_width < tt::constants::TILE_HEIGHT) ? remaining_width : tt::constants::TILE_HEIGHT;

            for (uint32_t local_w = 0; local_w < width_limit; local_w++) {
                const uint32_t patch_idx = row_patch_base + stride_w_idx;
                // Scatter each input pixel's C real bytes into its patch slot in the output stick.
                noc_async_write_sharded(
                    noc, src_base + local_w * c_padded_bytes, dst, out_page, patch_idx * c_bytes, c_bytes);
                if (++stride_w_idx == stride_width) {
                    stride_w_idx = 0;
                    out_page++;
                }
            }

            remaining_width -= tt::constants::TILE_HEIGHT;
            noc.async_write_barrier();
            dfb_in1.pop_front(tiles_per_channel_dim);
        }

        // Advance to the next output-h row only after `sh` input rows have been scattered.
        if (++patch_height_offset == stride_height) {
            curr_out_page += output_width;
            patch_height_offset = 0;
        }
    }
}
