// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "common_dataflow.hpp"

namespace {

FORCE_INLINE void zero_buffer(uint32_t write_addr, int bytes) {
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes > 0) {
        uint32_t curr_bytes = std::min(bytes, MEM_ZEROS_SIZE);
        noc_async_read(zeros_noc_addr, write_addr, curr_bytes);
        write_addr += curr_bytes;
        bytes -= curr_bytes;
    }
    noc_async_read_barrier();
}

template <typename input_number_t>
FORCE_INLINE void prepare_start_tile_for_cumsum_axis_2(uint32_t cb_start, uint32_t tile_size) {
    WriteCBGuard start_cb_guard{cb_start, ONE_TILE};

    uint32_t start_addr = get_write_ptr(cb_start);
    zero_buffer(start_addr, tile_size * sizeof(input_number_t));
}

template <typename input_addr_gen_t>
FORCE_INLINE void send_block(
    const input_addr_gen_t& input_addr_gen,
    uint32_t cb_input,
    uint32_t channels_slice_i,
    uint32_t column_block_i,
    uint32_t row_chunk_i,
    uint32_t num_blocks_in_column,
    uint32_t num_slices_along_channels,
    uint32_t block_depth) {
    for (uint32_t inner_tile_stride = 0; inner_tile_stride < block_depth; ++inner_tile_stride) {
        const uint32_t read_tile_id = get_tile_id(
            num_blocks_in_column,
            num_slices_along_channels,
            inner_tile_stride,
            channels_slice_i,
            row_chunk_i,
            column_block_i,
            block_depth);
        load_from_dram(cb_input, input_addr_gen, read_tile_id);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t input_base_addr = get_arg_val<uint32_t>(0);
    constexpr auto ctas = get_ctas();
    using input_number_type = std_type_t<get_dataformat(ctas.input_cb)>;
    const auto input_addr_gtor = TensorAccessor(ctas.input_args, input_base_addr, get_tile_size(ctas.input_cb));
    constexpr uint32_t num_slices_along_channels = ceil(ctas.num_channels, ctas.tile_width);
    constexpr uint32_t num_blocks_in_row = ceil(ctas.input_depth, ctas.block_depth);
    constexpr uint32_t num_blocks_in_column = ceil(ctas.input_height, ctas.tile_width);

    const auto core_x = get_absolute_logical_x();
    const auto core_y = get_absolute_logical_y();
    const uint32_t my_channel = core_y * ctas.cores_x + core_x;

    for (uint32_t batch_i = 0; batch_i < ctas.num_batches;
         ++batch_i) {  // only one batch expected, unit tests don't cover more, also not everything is implemented in
                       // terms of num_batches > 1
        for (uint32_t row_chunk_i = 0; row_chunk_i < num_blocks_in_column; ++row_chunk_i) {
            for (uint32_t column_block_i = 0; column_block_i < num_blocks_in_row; ++column_block_i) {
                prepare_start_tile_for_cumsum_axis_2<input_number_type>(
                    ctas.start_cb, ctas.tile_height * ctas.tile_width);
                const uint32_t block_depth =
                    std::min(ctas.input_depth - column_block_i * ctas.block_depth, ctas.block_depth);
                send_block(
                    input_addr_gtor,
                    ctas.input_cb,
                    my_channel,
                    column_block_i,
                    row_chunk_i,
                    num_blocks_in_column,
                    num_slices_along_channels,
                    block_depth);
            }
        }
    }
}
