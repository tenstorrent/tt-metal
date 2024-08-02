// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"


void kernel_main() {

    constexpr uint32_t cb_id_weight = get_compile_time_arg_val(0);
    constexpr uint32_t core_in_channels_ntiles = get_compile_time_arg_val(1);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(2);
    constexpr uint32_t weight_block_width_ntiles = get_compile_time_arg_val(3);
    constexpr uint32_t weight_block_num_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t weight_matrix_width_ntiles = get_compile_time_arg_val(5);
    constexpr uint32_t weight_next_channel_stride_h = get_compile_time_arg_val(6);
    constexpr uint32_t weight_next_block_this_core_stride_h = get_compile_time_arg_val(7);
    constexpr uint32_t weight_next_block_other_core_stride_h = get_compile_time_arg_val(8);
    constexpr uint32_t other_core_weight_height_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t this_core_weight_height_blocks = get_compile_time_arg_val(10);



    DPRINT<<"Weights  "<<cb_id_weight<<" "<<core_in_channels_ntiles<<" "<<window_size_hw<<" "<<weight_block_width_ntiles<<" "<<
    weight_block_num_tiles<<" "<<weight_matrix_width_ntiles<<" "<<weight_next_channel_stride_h<<" "<<weight_next_block_this_core_stride_h<<" "<<
    weight_next_block_other_core_stride_h<<"  "<<other_core_weight_height_blocks<<" "<<this_core_weight_height_blocks<<ENDL();

    uint32_t i = 0;
    uint32_t weight_start_tile_id = get_arg_val<uint32_t>(i); i+=1;
    const uint32_t weight_addr_dram_base = get_arg_val<uint32_t>(i); i+=1;

    const uint32_t weight_tile_nbytes = get_tile_size(cb_id_weight);
    const DataFormat weight_df = get_dataformat(cb_id_weight);

    const InterleavedAddrGenFast<true> s_weight = {
        .bank_base_address = weight_addr_dram_base,
        .page_size = weight_tile_nbytes,
        .data_format = weight_df
    };


    //Repeat for each weight block along width.
    for(uint32_t this_core_weight_block_index = 0; this_core_weight_block_index < this_core_weight_height_blocks; this_core_weight_block_index++)
    {
        uint32_t weight_block_start_tile_id = weight_start_tile_id;
        //repeat for each block that comes from each core.
        for(uint32_t other_core_weight_block_index = 0; other_core_weight_block_index < other_core_weight_height_blocks; other_core_weight_block_index++)
        {
            cb_reserve_back(cb_id_weight, weight_block_num_tiles);
            uint32_t weight_write_l1_addr = get_write_ptr(cb_id_weight);
            // mcast args
            uint32_t weights_start_address = weight_write_l1_addr;
            uint32_t weights_block_size_bytes = 0;
            uint32_t weight_current_block_start_tile_id = weight_block_start_tile_id;
            DPRINT<<"Read Start tile "<<weight_current_block_start_tile_id<<"\n";

            //for window size, picking up the channels for that window.
            for(uint32_t block_weight_h = 0; block_weight_h < window_size_hw; block_weight_h++) {
                uint32_t weight_row_start_tile_id = weight_current_block_start_tile_id;

                // mcast args
                //uint32_t weights_start_address = weight_write_l1_addr;
                //uint32_t weights_block_size_bytes = 0;

                // for number of input channels in one block
                for(uint32_t weight_tile_h_i = 0; weight_tile_h_i < core_in_channels_ntiles; ++weight_tile_h_i) { // TODO: 2
                    uint32_t weight_tile_id = weight_row_start_tile_id;

                    // loop over output channels
                    for(uint32_t weight_tile_w_i = 0; weight_tile_w_i < weight_block_width_ntiles; ++weight_tile_w_i) {
                        s_weight.noc_async_read_tile(weight_tile_id, weight_write_l1_addr);
                        weight_write_l1_addr += weight_tile_nbytes;
                        weights_block_size_bytes += weight_tile_nbytes;
                        weight_tile_id += 1;
                    } // for weight_block_w
                    weight_row_start_tile_id += weight_matrix_width_ntiles;
                } // for weight_block_h
                weight_current_block_start_tile_id += weight_next_channel_stride_h;
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_weight,weight_block_num_tiles);
            weight_block_start_tile_id += weight_next_block_other_core_stride_h;
        }
        weight_start_tile_id +=weight_next_block_this_core_stride_h;
    }
}
