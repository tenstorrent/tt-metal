// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

#include "common_dataflow.hpp"

namespace {

#ifdef ARCH_QUASAR
// Quasar's row-broadcast unpack only takes row 0 (bcast_row_idx must be 0), but the compute kernel adds the
// previous row block's last row (row 31). Read the tile back and copy row 31 over row 0 before handing it on.
// A 32x32 tile is four 16x16 faces in order (0 1 / 2 3): row 31 is row 15 of faces 2 and 3, row 0 is row 0 of
// faces 0 and 1.
template <typename addr_gen_type>
FORCE_INLINE void load_last_row_as_row_0(uint32_t cb, const addr_gen_type& addr_gtor, uint32_t read_tile_id) {
    const WriteCBGuard write_guard{cb, ONE_TILE};

    const Noc noc;
    const DataflowBuffer cb_obj(static_cast<uint16_t>(cb));
    noc.async_read(
        addr_gtor,
        cb_obj,
        addr_gtor.get_aligned_page_size(),
        {.page_id = read_tile_id, .offset_bytes = 0},
        {.offset_bytes = 0});
    noc.async_read_barrier();

    constexpr uint32_t FACE_ROWS = 16;
    const uint32_t face_words = cb_obj.get_entry_size() / 4 / sizeof(uint32_t);
    const uint32_t row_words = face_words / FACE_ROWS;
    volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_obj.get_write_ptr());
    for (uint32_t half = 0; half < 2; ++half) {
        volatile tt_l1_ptr uint32_t* dst = tile + half * face_words;
        volatile tt_l1_ptr uint32_t* src = tile + (2 + half) * face_words + (FACE_ROWS - 1) * row_words;
        for (uint32_t i = 0; i < row_words; ++i) {
            dst[i] = src[i];
        }
    }
}
#endif

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
#ifdef ARCH_QUASAR
        load_last_row_as_row_0(cb_axis_3_buffer_write, output_addr_gen, read_tile_id);
#else
        load_from_dram(cb_axis_3_buffer_write, output_addr_gen, read_tile_id, ONE_TILE);
#endif
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
    constexpr auto ctas = get_ctas();
    const auto output_addr_gtor = TensorAccessor(tensor::output);
    constexpr uint32_t num_slices_along_channels = ceil(ctas.num_channels, ctas.tile_width);
    constexpr uint32_t num_blocks_in_row = ceil(ctas.input_depth, ctas.block_depth);
    constexpr uint32_t num_blocks_in_column = ceil(ctas.input_height, ctas.tile_height);

    const auto core_x = get_absolute_logical_x();
    const auto core_y = get_absolute_logical_y();
    const uint32_t my_core = (core_y * ctas.cores_x) + core_x;

    const uint32_t num_cores = ctas.cores_x * ctas.cores_y;
    for (uint32_t my_channel = my_core; my_channel < num_slices_along_channels; my_channel += num_cores) {
        for (uint32_t batch_i = 0; batch_i < ctas.num_batches;
             ++batch_i) {  // only one batch expected, unit tests don't cover more, also not everything is implemented
                           // in terms of num_batches > 1
            for (uint32_t row_chunk_i = 0; row_chunk_i < num_blocks_in_column; ++row_chunk_i) {
                for (uint32_t column_block_i = 0; column_block_i < num_blocks_in_row; ++column_block_i) {
                    const uint32_t block_depth =
                        std::min(ctas.input_depth - (column_block_i * ctas.block_depth), ctas.block_depth);
                    if (row_chunk_i > 0) {
                        const uint32_t previous_row_chunk_i = row_chunk_i - 1;
                        receive_upper_block(
                            output_addr_gtor,
                            dfb::axis_3_buffer,
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
                        dfb::output,
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
}
