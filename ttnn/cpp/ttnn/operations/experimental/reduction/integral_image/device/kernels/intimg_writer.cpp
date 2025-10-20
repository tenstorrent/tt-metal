// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

#include "common.hpp"

#include <cmath>

namespace {

template <typename addr_gen_t>
FORCE_INLINE void write_to_dram(
    uint32_t cb, const addr_gen_t& addr_gtor, uint32_t write_tile_id, uint32_t num_tiles = ONE_TILE) {
    ReadCBGuard read_guard{cb, num_tiles};

    uint32_t l1_read_addr{get_read_ptr(cb)};
    noc_async_write_tile(write_tile_id, addr_gtor, l1_read_addr);
    noc_async_write_barrier();
}

template <typename addr_gen_type>
FORCE_INLINE void load_to_cb(
    uint32_t cb, const addr_gen_type& addr_gtor, uint32_t read_tile_id, uint32_t num_tiles = ONE_TILE) {
    WriteCBGuard write_guard{cb, num_tiles};

    uint32_t l1_write_addr{get_write_ptr(cb)};
    noc_async_read_tile(read_tile_id, addr_gtor, l1_write_addr);
    noc_async_read_barrier();
}

template <typename output_addr_gen_t>
FORCE_INLINE void receive_upper_block(
    const output_addr_gen_t& output_addr_gen,
    uint32_t cb_axis_3_buffer_write,
    uint32_t channels_slice_i,
    uint32_t column_block_i,
    uint32_t row_block_i,
    uint32_t num_blocks_in_row,
    uint32_t num_blocks_in_column,
    uint32_t num_slices_along_channels,
    uint32_t block_depth = 32) {
    for (uint32_t tile_i = 0; tile_i < block_depth; ++tile_i) {
        const uint32_t read_tile_id = get_tile_id(
            num_blocks_in_row,
            num_blocks_in_column,
            num_slices_along_channels,
            tile_i,
            channels_slice_i,
            row_block_i,
            column_block_i,
            block_depth);
        load_to_cb(cb_axis_3_buffer_write, output_addr_gen, read_tile_id, ONE_TILE);
    }
}

template <typename output_addr_gen_t>
FORCE_INLINE void output_block(
    const output_addr_gen_t& output_addr_gen,
    uint32_t cb_output,
    uint32_t channels_slice_i,
    uint32_t column_block_i,
    uint32_t row_block_i,
    uint32_t num_blocks_in_row,
    uint32_t num_blocks_in_column,
    uint32_t num_slices_along_channels,
    uint32_t block_depth = 32) {
    for (uint32_t inner_tile_stride = 0; inner_tile_stride < block_depth; ++inner_tile_stride) {
        const uint32_t write_tile_id = get_tile_id(
            num_blocks_in_row,
            num_blocks_in_column,
            num_slices_along_channels,
            inner_tile_stride,
            channels_slice_i,
            row_block_i,
            column_block_i,
            block_depth);
        write_to_dram(cb_output, output_addr_gen, write_tile_id);
    }
}

template <typename output_number_t, typename output_addr_gen_t>
FORCE_INLINE void broadcast_last_row_to_all_rows_in_cube(
    const output_addr_gen_t& output_addr_gen,
    uint32_t axis_3_propagation_read_cb,
    uint32_t axis_3_propagation_write_cb,
    uint32_t block_depth = 32) {
    // get the output after processing previous row (the block right above the currently processed block)
    // make the axis 3 propagation tile
    // push back to the compute
    constexpr uint32_t LAST_ROW_ORD = 32 - 1;  // tile_height - 1
    for (uint32_t tile_i = 0; tile_i < block_depth; ++tile_i) {
        ReadCBGuard propagation_upper_read_guard{axis_3_propagation_read_cb, ONE_TILE};
        WriteCBGuard propagation_upper_write_guard{axis_3_propagation_write_cb, ONE_TILE};
        uint32_t propagation_read_addr = get_read_ptr(axis_3_propagation_read_cb);
        uint32_t propagation_write_addr = get_write_ptr(axis_3_propagation_write_cb);
        volatile tt_l1_ptr output_number_t* propagation_read_ptr =
            reinterpret_cast<volatile tt_l1_ptr output_number_t*>(propagation_read_addr);
        volatile tt_l1_ptr output_number_t* propagation_write_ptr =
            reinterpret_cast<volatile tt_l1_ptr output_number_t*>(propagation_write_addr);
        for (uint32_t column_read_i = 0; column_read_i < 32; ++column_read_i) {
            output_number_t value_to_broadcast =
                propagation_read_ptr[get_coord_from_tile_xy(column_read_i, LAST_ROW_ORD)];
            for (uint32_t row_write_i = 0; row_write_i < 32; ++row_write_i) {
                propagation_write_ptr[get_coord_from_tile_xy(column_read_i, row_write_i)] = value_to_broadcast;
            }
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t output_base_addr = get_arg_val<uint32_t>(0);
    constexpr auto ctas = get_ctas();
    using output_number_type = std_type_t<get_dataformat(ctas.output_cb)>;
    const auto output_addr_gtor = TensorAccessor(ctas.output_args, output_base_addr, get_tile_size(ctas.output_cb));
    const uint32_t num_slices_along_channels = block_depth_ceil(
        ctas.num_channels, ctas.block_depth);  // block_depth is expected to be a power of 2 (the default is the regular
                                               // 32x32 tile's width/height size, that is, 32)
    const uint32_t num_blocks_in_row = block_depth_ceil(ctas.input_depth, ctas.block_depth);
    const uint32_t num_blocks_in_column = block_depth_ceil(ctas.input_height, ctas.block_depth);

    for (uint32_t batch_i = 0; batch_i < ctas.num_batches;
         ++batch_i) {  // only one batch expected, unit tests don't cover more, also not everything is implemented in
                       // terms of num_batches > 1
        for (uint32_t channels_slice_i = 0; channels_slice_i < num_slices_along_channels; ++channels_slice_i) {
            for (uint32_t row_chunk_i = 0; row_chunk_i < num_blocks_in_column; ++row_chunk_i) {
                for (uint32_t column_block_i = 0; column_block_i < num_blocks_in_row; ++column_block_i) {
                    const uint32_t block_depth =
                        std::min(ctas.input_depth - column_block_i * ctas.block_depth, ctas.block_depth);
                    if (row_chunk_i > 0) {
                        receive_upper_block(
                            output_addr_gtor,
                            ctas.axis_3_buffer_0_cb,
                            channels_slice_i,
                            column_block_i,
                            row_chunk_i - 1,
                            num_blocks_in_row,
                            num_blocks_in_column,
                            num_slices_along_channels,
                            block_depth);
                        broadcast_last_row_to_all_rows_in_cube<output_number_type, decltype(output_addr_gtor)>(
                            output_addr_gtor, ctas.axis_3_buffer_0_cb, ctas.axis_3_buffer_1_cb, block_depth);
                    }
                    output_block(
                        output_addr_gtor,
                        ctas.output_cb,
                        channels_slice_i,
                        column_block_i,
                        row_chunk_i,
                        num_blocks_in_row,
                        num_blocks_in_column,
                        num_slices_along_channels,
                        block_depth);
                }
            }
        }
    }
}
