// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

#include "common.hpp"

namespace {

template <typename output_addr_gen_t>
FORCE_INLINE void receive_whole_row(
    const output_addr_gen_t& output_addr_gen,
    uint32_t cb_output,
    uint32_t channels_slice_i,
    // uint32_t column_block_i,
    uint32_t row_chunk_i,
    // uint32_t num_blocks_in_row,
    uint32_t input_depth,
    uint32_t num_tiles_in_column,
    uint32_t num_tiles_along_channels) {
    for (uint32_t row_tile_stride = 0; row_tile_stride < input_depth; ++row_tile_stride) {
        const uint32_t write_tile_id =
            get_tile_id(num_tiles_in_column, num_tiles_along_channels, row_tile_stride, channels_slice_i, row_chunk_i);
        write_to_dram(cb_output, output_addr_gen, write_tile_id);
    }
}

// template <typename output_number_t>
// FORCE_INLINE void broadcast_last_row_of_upper_tile_to_all_rows_in_current_tile(
//     uint32_t axis_3_propagation_read_cb, uint32_t axis_3_propagation_write_cb) {
//     ReadCBGuard propagation_upper_read_guard{axis_3_propagation_read_cb, ONE_TILE};
//     WriteCBGuard propagation_upper_write_guard{axis_3_propagation_write_cb, ONE_TILE};
//     uint32_t propagation_read_addr = get_read_ptr(axis_3_propagation_read_cb);
//     uint32_t propagation_write_addr = get_write_ptr(axis_3_propagation_write_cb);
//     volatile tt_l1_ptr output_number_t* propagation_read_ptr =
//         reinterpret_cast<volatile tt_l1_ptr output_number_t*>(propagation_read_addr);
//     volatile tt_l1_ptr output_number_t* propagation_write_ptr =
//         reinterpret_cast<volatile tt_l1_ptr output_number_t*>(propagation_write_addr);
//     for (uint32_t column_read_i = 0; column_read_i < 32; ++column_read_i) {  // TODO(jbbieniekTT): no magic in code!
//     ()
//         for (uint32_t row_write_i = 0; row_write_i < 32; ++row_write_i) {
//             propagation_write_ptr[get_coord_from_tile_xy(column_read_i, row_write_i)] =
//                 propagation_read_ptr[get_coord_from_tile_xy(column_read_i, LAST_ROW_ORD)];
//         }
//     }
// }

FORCE_INLINE void pass_tile_to_lower_core(
    uint32_t lower_core_x,
    uint32_t lower_core_y,
    uint32_t my_bot_semaphore_id,
    uint32_t their_top_semaphore_id,
    uint32_t cb_local_data_in,
    uint32_t cb_remote_data_out) {
    const uint64_t their_top_semaphore_addr = get_noc_addr(lower_core_x, lower_core_y, their_top_semaphore_id);
    volatile tt_l1_ptr uint32_t* my_bot_semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(my_bot_semaphore_id);

    ReadCBGuard cb_local_data_in_guard{cb_local_data_in, ONE_TILE};

    noc_semaphore_wait(my_bot_semaphore_ptr, SEMAPHORE_READY);
    const auto cb_remote_data_out_ptr = get_write_ptr(cb_remote_data_out);
    const uint64_t their_cb_write_addr = get_noc_addr(lower_core_x, lower_core_y, cb_remote_data_out_ptr);
    const auto cb_local_data_in_ptr = get_read_ptr(cb_local_data_in);

    noc_async_write(cb_local_data_in_ptr, their_cb_write_addr, get_tile_size(cb_remote_data_out));
    noc_async_write_barrier();

    noc_semaphore_inc(their_top_semaphore_addr, SEMAPHORE_READY);
    noc_async_atomic_barrier();
    noc_semaphore_set(my_bot_semaphore_ptr, SEMAPHORE_BUSY);
}

}  // namespace

void kernel_main() {
    const auto rtas = get_rtas();
    constexpr auto ctas = get_ctas();
    // using output_number_type = std_type_t<get_dataformat(ctas.output_cb)>;
    const auto output_addr_gtor = TensorAccessor(ctas.output_args, rtas.output_base_addr, get_tile_size(ctas.input_cb));
    const uint32_t top_semaphore_id = get_semaphore(ctas.top_semaphore);
    const uint32_t bot_semaphore_id = get_semaphore(ctas.bot_semaphore);

    // 90 DEGREES FLIP!!!!!!!!!!!!111111ONEONEONEONEONEONEONEONOENONEONEON
    const uint32_t core_x = get_absolute_logical_x();  // from 0 up to 4 (5 cores) - treated as "vertical"!
    const uint32_t upper_core_x = core_x - 1;
    const uint32_t lower_core_x = core_x + 1;
    const uint32_t core_y = get_absolute_logical_y();  // from 0 up to 3 (4 cores) - treated as "horizontal"!
    const uint32_t upper_core_y = core_y;
    const uint32_t lower_core_y = core_y;
    const uint32_t ending_tile_along_channels =
        rtas.starting_tile_along_channels + rtas.num_tiles_along_channels_per_core;
    const uint32_t ending_tile_along_height = rtas.starting_tile_along_height + rtas.num_tiles_along_height_per_core;
    for (uint32_t channels_slice_i = rtas.starting_tile_along_channels; channels_slice_i < ending_tile_along_channels;
         ++channels_slice_i) {
        for (uint32_t column_tile_i = rtas.starting_tile_along_height; column_tile_i < ending_tile_along_height;
             ++column_tile_i) {
            for (uint32_t row_tile_i = 0; row_tile_i < ctas.input_depth; ++row_tile_i) {
                const uint32_t write_tile_id = get_tile_id(
                    ctas.num_tiles_along_height,
                    ctas.num_tiles_along_channels,
                    row_tile_i,
                    channels_slice_i,
                    column_tile_i);

                if (column_tile_i < ctas.num_tiles_along_height - 1) {
                    pass_tile_to_lower_core(
                        lower_core_x,
                        lower_core_y,
                        bot_semaphore_id,
                        top_semaphore_id,
                        ctas.to_bot_tile_cb,
                        ctas.axis_3_buffer_0_cb);
                }
                write_to_dram(ctas.output_cb, output_addr_gtor, write_tile_id, ONE_TILE);
            }
        }
    }
}
