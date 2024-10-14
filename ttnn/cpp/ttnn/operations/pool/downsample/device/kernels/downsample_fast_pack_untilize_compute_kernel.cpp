// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tilize.h"

namespace NAMESPACE {
void MAIN {


    uint32_t i = 0;

    uint32_t local_input_num_rows_of_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t local_input_offset_rows_of_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_prev_enabled = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_prev_input_num_rows_of_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_next_enabled = get_arg_val<uint32_t>(i); i+=1;
    uint32_t halo_next_input_num_rows_of_tiles = get_arg_val<uint32_t>(i); i+=1;

    uint32_t input_cb_index = get_compile_time_arg_val(0); // sharded tiled
    uint32_t halo_prev_input_cb_index = get_compile_time_arg_val(1); // halo from prev core data tiled
    uint32_t halo_next_input_cb_index = get_compile_time_arg_val(2); // halo from next core data tiled
    uint32_t untilize_cb_index = get_compile_time_arg_val(3); // 1 row of tiles (untilized)
    uint32_t untilize_downsampled_cb_index = get_compile_time_arg_val(4); // full output size
    uint32_t tilize_out_cb_index = get_compile_time_arg_val(5); // final output = sharded output
    constexpr uint32_t num_input_tiles_in_row = get_compile_time_arg_val(6); // same for both local and halo inputs

    uint32_t num_output_rows_of_tiles = get_compile_time_arg_val(7);
    uint32_t num_output_tiles_in_row = get_compile_time_arg_val(8);
    uint32_t num_output_tiles = num_output_rows_of_tiles * num_output_tiles_in_row;

    //untilize_init(input_cb_index, untilize_cb_index);
    pack_untilize_init<num_input_tiles_in_row>(input_cb_index, untilize_cb_index);
    pack_reconfig_data_format(untilize_cb_index);

    // print_cb_details(input_cb_index);
    // print_cb_details(untilize_cb_index);
    // print_cb_details(untilize_downsampled_cb_index);
    // print_cb_details(tilize_out_cb_index);

    if (halo_prev_enabled) {
        // not required since cb has same data format and untilize init already configured unpacker
        //untilize_init_short(halo_prev_input_cb_index);

        // Untilize halo input
        for(uint32_t b = 0; b < halo_prev_input_num_rows_of_tiles; ++ b) {

            cb_wait_front(halo_prev_input_cb_index, num_input_tiles_in_row);
            cb_reserve_back(untilize_cb_index, num_input_tiles_in_row);

            pack_untilize_block<num_input_tiles_in_row>(halo_prev_input_cb_index, 1, untilize_cb_index);

            cb_push_back(untilize_cb_index, num_input_tiles_in_row);
            cb_pop_front(halo_prev_input_cb_index, num_input_tiles_in_row);
        }
    }

    // Unilize prev core's halo region - 1 row of tiles
    // untilize_block(prev_core_input_cb_index, num_input_tiles_per_block, untilize_cb_index);

    // Untilize input
    cb_pop_front(input_cb_index, local_input_offset_rows_of_tiles * num_input_tiles_in_row);
    for(uint32_t b = 0; b < local_input_num_rows_of_tiles; ++ b) {

        cb_reserve_back(untilize_cb_index, num_input_tiles_in_row);

        pack_untilize_block<num_input_tiles_in_row>(input_cb_index, 1, untilize_cb_index);

        cb_push_back(untilize_cb_index, num_input_tiles_in_row);
        cb_pop_front(input_cb_index, num_input_tiles_in_row);
    }

    if (halo_next_enabled) {
        // not required since cb has same data format and untilize init already configured unpacker
        //untilize_init_short(halo_next_input_cb_index);

        // Untilize halo input
        for(uint32_t b = 0; b < halo_next_input_num_rows_of_tiles; ++ b) {

            cb_wait_front(halo_next_input_cb_index, num_input_tiles_in_row);
            cb_reserve_back(untilize_cb_index, num_input_tiles_in_row);

            pack_untilize_block<num_input_tiles_in_row>(halo_next_input_cb_index, 1, untilize_cb_index);

            cb_push_back(untilize_cb_index, num_input_tiles_in_row);
            cb_pop_front(halo_next_input_cb_index, num_input_tiles_in_row);
        }
    }
    pack_untilize_uninit();

    // Tilize downsampled input
    cb_wait_front(untilize_downsampled_cb_index, num_output_tiles);
    cb_reserve_back(tilize_out_cb_index, num_output_tiles);

    reconfig_data_format_srca(input_cb_index, untilize_downsampled_cb_index);
    tilize_init_short(untilize_downsampled_cb_index, num_output_tiles_in_row);
    pack_reconfig_data_format(tilize_out_cb_index);

    for(uint32_t b=0;b<num_output_rows_of_tiles;++b)
    {
        cb_wait_front(untilize_downsampled_cb_index, num_output_tiles_in_row);
        cb_reserve_back(tilize_out_cb_index, num_output_tiles_in_row);

        tilize_block(untilize_downsampled_cb_index, num_output_tiles_in_row, tilize_out_cb_index);

        cb_push_back(tilize_out_cb_index, num_output_tiles_in_row);
        cb_pop_front(untilize_downsampled_cb_index, num_output_tiles_in_row);
    }
}
}
