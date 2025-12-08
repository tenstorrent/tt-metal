// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "common_dataflow.hpp"

namespace {

template <typename output_addr_gen_t>
FORCE_INLINE void receive_upper_block(
    const output_addr_gen_t& output_addr_gen,
    uint32_t cb_axis_3_buffer_write,
    uint32_t channels_slice_i,
    uint32_t column_block_i,
    uint32_t row_block_i,
    uint32_t num_blocks_in_column,
    uint32_t num_slices_along_channels,
    uint32_t block_depth,
    uint32_t generic_block_depth) {
    for (uint32_t tile_i = 0; tile_i < block_depth; ++tile_i) {
        const uint32_t read_tile_id = get_tile_id(
            num_blocks_in_column,
            num_slices_along_channels,
            tile_i,
            channels_slice_i,
            row_block_i,
            column_block_i,
            generic_block_depth);
        load_from_dram(cb_axis_3_buffer_write, output_addr_gen, read_tile_id, ONE_TILE);
    }
}

template <typename output_addr_gen_t>
FORCE_INLINE void output_block(
    const output_addr_gen_t& output_addr_gen,
    uint32_t cb_output,
    uint32_t channels_slice_i,
    uint32_t column_block_i,
    uint32_t row_block_i,
    uint32_t num_blocks_in_column,
    uint32_t num_slices_along_channels,
    uint32_t block_depth,
    uint32_t generic_block_depth) {
    for (uint32_t inner_tile_stride = 0; inner_tile_stride < block_depth; ++inner_tile_stride) {
        const uint32_t write_tile_id = get_tile_id(
            num_blocks_in_column,
            num_slices_along_channels,
            inner_tile_stride,
            channels_slice_i,
            row_block_i,
            column_block_i,
            generic_block_depth);
        write_to_dram(cb_output, output_addr_gen, write_tile_id);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t output_base_addr = get_arg_val<uint32_t>(0);
    constexpr auto ctas = get_ctas();
    const auto output_addr_gtor = TensorAccessor(ctas.output_args, output_base_addr, get_tile_size(ctas.output_cb));
    constexpr uint32_t num_slices_along_channels = ceil(ctas.num_channels, ctas.tile_width);
    constexpr uint32_t num_blocks_in_row = ceil(ctas.input_depth, ctas.block_depth);
    constexpr uint32_t num_blocks_in_column = ceil(ctas.input_height, ctas.tile_height);

    const auto core_x = get_absolute_logical_x();
    const auto core_y = get_absolute_logical_y();
    const uint32_t my_channel = core_y * ctas.cores_x + core_x;

    for (uint32_t batch_i = 0; batch_i < ctas.num_batches;
         ++batch_i) {  // only one batch expected, unit tests don't cover more, also not everything is implemented in
                       // terms of num_batches > 1
        for (uint32_t row_chunk_i = 0; row_chunk_i < num_blocks_in_column; ++row_chunk_i) {
            for (uint32_t column_block_i = 0; column_block_i < num_blocks_in_row; ++column_block_i) {
                const uint32_t block_depth =
                    std::min(ctas.input_depth - column_block_i * ctas.block_depth, ctas.block_depth);
                if (row_chunk_i > 0) {
                    const uint32_t previous_row_chunk_i = row_chunk_i - 1;
                    receive_upper_block(
                        output_addr_gtor,
                        ctas.axis_3_buffer_cb,
                        my_channel,
                        column_block_i,
                        previous_row_chunk_i,
                        num_blocks_in_column,
                        num_slices_along_channels,
                        block_depth,
                        ctas.block_depth);
                }
                output_block(
                    output_addr_gtor,
                    ctas.output_cb,
                    my_channel,
                    column_block_i,
                    row_chunk_i,
                    num_blocks_in_column,
                    num_slices_along_channels,
                    block_depth,
                    ctas.block_depth);
            }
        }
    }
}
