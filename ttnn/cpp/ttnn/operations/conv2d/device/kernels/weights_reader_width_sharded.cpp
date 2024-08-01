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
    constexpr uint32_t weight_next_block_stride_h = get_compile_time_arg_val(7);
    constexpr uint32_t weight_num_height_blocks = get_compile_time_arg_val(8);



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


    uint32_t weight_block_start_tile_id = weight_start_tile_id;

    for(uint32_t weight_tile_h_outer_i = 0; weight_tile_h_outer_i < weight_num_height_blocks; weight_tile_h_outer_i++)
    {
        cb_reserve_back(cb_id_weight, weight_block_num_tiles);
        uint32_t weight_write_l1_addr = get_write_ptr(cb_id_weight);
        // mcast args
        uint32_t weights_start_address = weight_write_l1_addr;
        uint32_t weights_block_size_bytes = 0;
        uint32_t weight_current_block_start_tile_id = weight_block_start_tile_id;

        for(uint32_t block_weight_h = 0; block_weight_h < window_size_hw; block_weight_h++) { // TODO: 9
            uint32_t weight_row_start_tile_id = weight_current_block_start_tile_id;

            // mcast args
            //uint32_t weights_start_address = weight_write_l1_addr;
            //uint32_t weights_block_size_bytes = 0;

            // loop over weight block tiles along h
            for(uint32_t weight_tile_h_i = 0; weight_tile_h_i < core_in_channels_ntiles; ++weight_tile_h_i) { // TODO: 2
                uint32_t weight_tile_id = weight_row_start_tile_id;
                // loop over weight block tiles along w
                for(uint32_t weight_tile_w_i = 0; weight_tile_w_i < weight_block_width_ntiles; ++weight_tile_w_i) {
                    // DPRINT<<weight_tile_id<<"\n";
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
        weight_block_start_tile_id +=weight_next_block_stride_h;
    }
}
