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

    const uint32_t start_super_block_id = get_arg(args::start_block_id);
    const uint32_t num_super_blocks = get_arg(args::num_blocks);

    constexpr uint32_t output_width = input_width / stride_width;
    constexpr uint32_t patch_size = stride_height * stride_width;
    constexpr uint32_t output_stick_bytes = patch_size * c_bytes;

    const auto dst = TensorAccessor(tensor::dst);
    Noc noc;
    DataflowBuffer dfb_untilized(dfb::in1);
    DataflowBuffer dfb_scratch(dfb::in2);

    // cb_asm holds one full output row; touched only by this kernel, so raw pointer + local ordering.
    const uint32_t scratch_base = dfb_scratch.get_write_ptr();

    const uint32_t end_super_block_id = start_super_block_id + num_super_blocks;
    for (uint32_t sb = start_super_block_id; sb < end_super_block_id; ++sb) {
        // Gather `stride_height` input rows worth of C-byte sticks into cb_asm; each pixel lands at
        // `out_w * output_stick_bytes + patch_idx * c_bytes` so every output stick becomes contiguous.
        for (uint32_t local_h = 0; local_h < stride_height; ++local_h) {
            uint32_t remaining_width = input_width;
            const uint32_t row_patch_base = local_h * stride_width;
            for (uint32_t w_tile = 0; w_tile < tiles_per_width_dim; ++w_tile) {
                dfb_untilized.wait_front(tiles_per_channel_dim);
                const uint32_t src_base = dfb_untilized.get_read_ptr();

                const uint32_t width_limit =
                    (remaining_width < tt::constants::TILE_HEIGHT) ? remaining_width : tt::constants::TILE_HEIGHT;
                const uint32_t w_base = w_tile * tt::constants::TILE_HEIGHT;
                for (uint32_t local_w = 0; local_w < width_limit; ++local_w) {
                    const uint32_t w_in = w_base + local_w;
                    const uint32_t out_w = w_in / stride_width;
                    const uint32_t patch_idx = row_patch_base + (w_in % stride_width);
                    tt_memmove<false, false, false, c_bytes>(
                        noc,
                        scratch_base + out_w * output_stick_bytes + patch_idx * c_bytes,
                        src_base + local_w * c_padded_bytes,
                        c_bytes);
                }
                remaining_width -= tt::constants::TILE_HEIGHT;
                dfb_untilized.pop_front(tiles_per_channel_dim);
            }
        }
        // Emit one aligned page-sized write per output stick; super-blocks are laid out sequentially.
        const uint32_t output_page_base = sb * output_width;
        for (uint32_t out_w = 0; out_w < output_width; ++out_w) {
            noc_async_write_sharded(
                noc,
                scratch_base + out_w * output_stick_bytes,
                dst,
                output_page_base + out_w,
                /*offset=*/0,
                output_stick_bytes);
        }
        noc.async_write_barrier();
    }
}
