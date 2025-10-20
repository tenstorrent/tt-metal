// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "common.hpp"

#include "debug/dprint.h"

namespace {

template <typename input_number_t>
FORCE_INLINE void prepare_start_tile_for_cumsum_axis_2(uint32_t cb_start, uint32_t tile_size) {
    // TODO(jbbieniek): consider Saad's method of nullifying CB's contents
    WriteCBGuard start_cb_guard{cb_start, ONE_TILE};

    uint32_t start_addr = get_write_ptr(cb_start);
    volatile tt_l1_ptr input_number_t* start_ptr = reinterpret_cast<volatile tt_l1_ptr input_number_t*>(start_addr);
    for (uint32_t i = 0; i < tile_size; i++) {
        start_ptr[i] = 0;
    }
}

template <typename addr_gen_type>
FORCE_INLINE void load_to_cb(
    uint32_t cb, const addr_gen_type& addr_gtor, uint32_t read_tile_id, uint32_t num_tiles = ONE_TILE) {
    WriteCBGuard write_guard{cb, num_tiles};

    uint32_t l1_write_addr{get_write_ptr(cb)};
    noc_async_read_tile(read_tile_id, addr_gtor, l1_write_addr);
    noc_async_read_barrier();
}

template <typename input_addr_gen_t>
FORCE_INLINE void send_block(
    const input_addr_gen_t& input_addr_gen,
    uint32_t cb_input,
    uint32_t channels_slice_i,
    uint32_t column_block_i,
    uint32_t row_chunk_i,
    uint32_t num_blocks_in_row,
    uint32_t num_blocks_in_column,
    uint32_t num_slices_along_channels,
    uint32_t block_depth = 32) {
    for (uint32_t inner_tile_stride = 0; inner_tile_stride < block_depth; ++inner_tile_stride) {
        const uint32_t read_tile_id = get_tile_id(
            num_blocks_in_row,
            num_blocks_in_column,
            num_slices_along_channels,
            inner_tile_stride,
            channels_slice_i,
            row_chunk_i,
            column_block_i,
            block_depth);
        // DPRINT << "sending tile id " << read_tile_id << ENDL();
        load_to_cb(cb_input, input_addr_gen, read_tile_id);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t input_base_addr = get_arg_val<uint32_t>(0);
    constexpr auto ctas = get_ctas();
    using input_number_type = std_type_t<get_dataformat(ctas.input_cb)>;
    // constexpr uint32_t tensor_size = ctas.num_batches * ctas.input_depth * 32 * 32;
    const auto input_addr_gtor = TensorAccessor(ctas.input_args, input_base_addr, get_tile_size(ctas.input_cb));
    const uint32_t num_slices_along_channels = block_depth_ceil(
        ctas.num_channels, ctas.block_depth);  // block_depth is expected to be a power of 2 (the default is the regular
                                               // 32x32 tile's width/height size, that is, 32)
    const uint32_t num_blocks_in_row = block_depth_ceil(ctas.input_depth, ctas.block_depth);
    const uint32_t num_blocks_in_column = block_depth_ceil(ctas.input_height, ctas.block_depth);
    DPRINT << "channel blocks: " << num_slices_along_channels << ", row blocks: " << num_blocks_in_row
           << ", column_blocks: " << num_blocks_in_column << ENDL();

    for (uint32_t batch_i = 0; batch_i < ctas.num_batches;
         ++batch_i) {  // only one batch expected, unit tests don't cover more, also not everything is implemented in
                       // terms of num_batches > 1
        for (uint32_t channels_slice_i = 0; channels_slice_i < num_slices_along_channels; ++channels_slice_i) {
            for (uint32_t row_chunk_i = 0; row_chunk_i < num_blocks_in_column; ++row_chunk_i) {
                for (uint32_t column_block_i = 0; column_block_i < num_blocks_in_row; ++column_block_i) {
                    prepare_start_tile_for_cumsum_axis_2<input_number_type>(
                        ctas.start_cb, ctas.tile_height * ctas.tile_width);
                    const uint32_t block_depth =
                        std::min(ctas.input_depth - column_block_i * ctas.block_depth, ctas.block_depth);
                    send_block(
                        input_addr_gtor,
                        ctas.input_cb,
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
