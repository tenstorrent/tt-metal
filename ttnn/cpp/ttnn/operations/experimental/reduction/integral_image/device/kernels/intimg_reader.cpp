// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "common.hpp"

#include "debug/dprint.h"

namespace {

// template <typename zero_tile_addr_gen_t>
// FORCE_INLINE void prepare_start_tile_for_cumsum(uint32_t cb_start, const zero_tile_addr_gen_t& zero_tile_addr_gtor) {
template <typename input_number_type>
FORCE_INLINE void prepare_start_tile_for_cumsum(
    uint32_t cb_start, uint32_t tile_height = 32, uint32_t tile_width = 32) {
    WriteCBGuard start_cb_guard{cb_start, ONE_TILE};

    const uint32_t tile_elements = tile_height * tile_width;
    uint32_t start_addr = get_write_ptr(cb_start);
    volatile tt_l1_ptr input_number_type* start_ptr =
        reinterpret_cast<volatile tt_l1_ptr input_number_type*>(start_addr);
    for (uint32_t i = 0; i < tile_elements; i++) {
        start_ptr[i] = 0;
    }
    // load_to_cb(cb_start, zero_tile_addr_gtor, FIRST_TILE, ONE_TILE);
}

template <typename input_addr_gen_t>
FORCE_INLINE void pass_input_tile_to_compute(
    const input_addr_gen_t& input_addr_gen,
    uint32_t cb_input,
    uint32_t channels_tile_i,
    uint32_t column_tile_i,
    uint32_t row_tile_stride,
    uint32_t num_tiles_in_column,
    uint32_t num_tiles_along_channels) {
    // for (uint32_t row_tile_stride = 0; row_tile_stride < input_depth; ++row_tile_stride) {
    const uint32_t read_tile_id =
        get_tile_id(num_tiles_in_column, num_tiles_along_channels, row_tile_stride, channels_tile_i, column_tile_i);
    load_to_cb(cb_input, input_addr_gen, read_tile_id);
    // DPRINT << "inner_tile_stride/block_depth: " << inner_tile_stride << "/" << block_depth << ENDL();
    // DPRINT << "sending tile id " << read_tile_id << ENDL();
    // }
}

FORCE_INLINE void get_tile_from_upper_core_writer(
    uint32_t upper_core_x,
    uint32_t upper_core_y,
    uint32_t my_top_semaphore_id,
    uint32_t their_bot_semaphore_id,
    uint32_t cb_local_data_in) {
    volatile tt_l1_ptr uint32_t* my_top_semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(my_top_semaphore_id);  // my semaphore ptr

    WriteCBGuard cb_local_data_in_guard{cb_local_data_in, ONE_TILE};

    uint64_t their_bot_semaphore_address =
        get_noc_addr(upper_core_x, upper_core_y, their_bot_semaphore_id);  // upper semaphore addr
    noc_semaphore_inc(their_bot_semaphore_address, SEMAPHORE_READY);       // inform them im ready to get a tile
    noc_async_atomic_barrier();                                            // commit and sync

    noc_semaphore_wait(
        my_top_semaphore_ptr, SEMAPHORE_READY);               // wait until they inform me they are ready to send a tile
    noc_semaphore_set(my_top_semaphore_ptr, SEMAPHORE_BUSY);  // reset my semaphore
};

template <typename input_number_t>
FORCE_INLINE void spawn_adder_tile_from_broadcasting_last_tile_and_pass_to_compute(
    uint32_t axis_3_propagation_read_cb,
    uint32_t axis_3_propagation_write_cb,
    uint32_t tile_height = 32,
    uint32_t tile_width = 32) {
    ReadCBGuard propagation_upper_read_guard{axis_3_propagation_read_cb, ONE_TILE};
    WriteCBGuard propagation_upper_write_guard{axis_3_propagation_write_cb, ONE_TILE};
    uint32_t propagation_read_addr = get_read_ptr(axis_3_propagation_read_cb);
    uint32_t propagation_write_addr = get_write_ptr(axis_3_propagation_write_cb);
    volatile tt_l1_ptr input_number_t* propagation_read_ptr =
        reinterpret_cast<volatile tt_l1_ptr input_number_t*>(propagation_read_addr);
    volatile tt_l1_ptr input_number_t* propagation_write_ptr =
        reinterpret_cast<volatile tt_l1_ptr input_number_t*>(propagation_write_addr);
    for (uint32_t column_read_i = 0; column_read_i < tile_width; ++column_read_i) {
        for (uint32_t row_write_i = 0; row_write_i < tile_height; ++row_write_i) {
            propagation_write_ptr[get_coord_from_tile_xy(column_read_i, row_write_i)] =
                propagation_read_ptr[get_coord_from_tile_xy(column_read_i, LAST_ROW_ORD)];
        }
    }
}

}  // namespace

void kernel_main() {
    const auto rtas = get_rtas();
    constexpr auto ctas = get_ctas();
    using input_number_type = std_type_t<get_dataformat(ctas.input_cb)>;
    const auto input_addr_gtor = TensorAccessor(ctas.input_args, rtas.input_base_addr, get_tile_size(ctas.input_cb));
    // const auto zero_tile_addr_gtor =
    //     TensorAccessor(ctas.zero_tile_args, rtas.zero_tile_base_addr, get_tile_size(ctas.input_cb));
    const uint32_t top_semaphore_id = get_semaphore(ctas.top_semaphore);
    const uint32_t bot_semaphore_id = get_semaphore(ctas.bot_semaphore);

    // 90 DEGREES FLIP!!!!!!!!!!!!111111ONEONEONEONEONEONEONEONOENONEONEON
    const uint32_t core_x = get_absolute_logical_x();  // from 0 up to 4 (5 cores) - treated as "vertical"!!!!!!!
    const uint32_t upper_core_x = core_x - 1;
    const uint32_t lower_core_x = core_x + 1;
    const uint32_t core_y = get_absolute_logical_y();  // from 0 up to 3 (4 cores) - treated as "horizontal"!!!!!!
    const uint32_t upper_core_y = core_y;
    const uint32_t lower_core_y = core_y;
    const uint32_t ending_tile_along_channels =
        rtas.starting_tile_along_channels + rtas.num_tiles_along_channels_per_core;
    const uint32_t ending_tile_along_height = rtas.starting_tile_along_height + rtas.num_tiles_along_height_per_core;
    for (uint32_t channels_tile_i = rtas.starting_tile_along_channels; channels_tile_i < ending_tile_along_channels;
         ++channels_tile_i) {
        for (uint32_t column_tile_i = rtas.starting_tile_along_height; column_tile_i < ending_tile_along_height;
             ++column_tile_i) {
            // produce: start_cb
            // prepare_start_tile_for_cumsum(ctas.start_cb, zero_tile_addr_gtor);
            prepare_start_tile_for_cumsum<input_number_type>(ctas.start_cb);
            for (uint32_t row_tile_i = 0; row_tile_i < ctas.input_depth; ++row_tile_i) {
                pass_input_tile_to_compute(
                    input_addr_gtor,
                    ctas.input_cb,
                    channels_tile_i,
                    column_tile_i,
                    row_tile_i,
                    ctas.num_tiles_along_height,
                    ctas.num_tiles_along_channels);
                if (column_tile_i > 0) {
                    get_tile_from_upper_core_writer(
                        upper_core_x, upper_core_y, top_semaphore_id, bot_semaphore_id, ctas.axis_3_buffer_0_cb);
                    spawn_adder_tile_from_broadcasting_last_tile_and_pass_to_compute<input_number_type>(
                        ctas.axis_3_buffer_0_cb, ctas.axis_3_buffer_1_cb, ctas.tile_height, ctas.tile_width);
                    // broadcast_last_row_of_upper_tile_to_all_rows_of_current_adder_tile();
                    // pass_gen2_tile_to_compute();
                }
                // if (row_chunk_i > 0) {
                //     get_tile_from_upper_core_and_pass_to_writer();
                //     // pass_tile_from_upper_core_to_writer();
                // }
            }
        }
    }
}
