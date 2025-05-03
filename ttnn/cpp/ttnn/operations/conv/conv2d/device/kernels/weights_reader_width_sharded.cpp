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

    // weight_block_width_ntiles corresponds to the full output width of each core.
    constexpr uint32_t weight_block_width_ntiles = get_compile_time_arg_val(3);
    constexpr uint32_t weight_block_num_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t weight_matrix_width_ntiles = get_compile_time_arg_val(5);
    constexpr uint32_t weight_next_channel_stride_h = get_compile_time_arg_val(6);
    constexpr uint32_t weight_next_block_this_core_stride_h = get_compile_time_arg_val(7);
    constexpr uint32_t weight_next_block_other_core_stride_h = get_compile_time_arg_val(8);
    constexpr uint32_t remote_weight_height_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t local_weight_height_blocks = get_compile_time_arg_val(10);
    constexpr uint32_t act_num_blocks_h = get_compile_time_arg_val(11);

#ifdef FUSE_BIAS
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(12);
    constexpr uint32_t bias_in_dram = get_compile_time_arg_val(13) == 1;
    constexpr bool has_bias = true;
#else
    constexpr bool has_bias = false;
#endif

    uint32_t i = 0;
    const uint32_t init_weight_start_tile_id = get_arg_val<uint32_t>(i);
    i += 1;
    const uint32_t weight_addr_dram_base = get_arg_val<uint32_t>(i);
    i += 1;

    uint32_t bias_addr_dram_base = get_arg_val<uint32_t>(i);
    i += 1;
    const uint32_t is_active = get_arg_val<uint32_t>(i);
    i += 1;

    const uint32_t weight_tile_nbytes = get_tile_size(cb_id_weight);
    const DataFormat weight_df = get_dataformat(cb_id_weight);

    const InterleavedAddrGenFast<true> s_weight = {
        .bank_base_address = weight_addr_dram_base, .page_size = weight_tile_nbytes, .data_format = weight_df};
#ifdef FUSE_BIAS
    const uint32_t bias_pagesize = get_tile_size(bias_cb_id);
    const DataFormat bias_df = get_dataformat(bias_cb_id);
    const InterleavedAddrGenFast<bias_in_dram> s_bias = {
        .bank_base_address = bias_addr_dram_base, .page_size = bias_pagesize, .data_format = bias_df};

#endif
    bool to_load_bias = true;

    for (uint32_t act_block_h_index = 0; act_block_h_index < act_num_blocks_h; act_block_h_index++) {
        uint32_t weight_start_tile_id = init_weight_start_tile_id;
        uint32_t bias_start_tile_id = init_weight_start_tile_id;

        // Outer most loop is each core's block width.
        // This interleaves reader/tilization with compute. Hopefully better perf.
        // Activation reader first sends data from the same block of all cores. Then iterates to the next block and
        // repeats for all cores. Stride = act_block_w*out_channels*sizeof(elem)
        for (uint32_t local_weight_block_index = 0; local_weight_block_index < local_weight_height_blocks;
             local_weight_block_index++) {
            uint32_t weight_block_start_tile_id = weight_start_tile_id;

            // Iterates over all the cores.
            // Stride = in_channels*out_channels*sizeof(elem)/num_cores.
            for (uint32_t remote_weight_block_index = 0; remote_weight_block_index < remote_weight_height_blocks;
                 remote_weight_block_index++) {
                cb_reserve_back(cb_id_weight, weight_block_num_tiles);
                if (is_active) {
                    uint32_t weight_write_l1_addr = get_write_ptr(cb_id_weight);
                    uint32_t weights_start_address = weight_write_l1_addr;
                    uint32_t weights_block_size_bytes = 0;
                    uint32_t weight_current_block_start_tile_id = weight_block_start_tile_id;

                    // for window size, picking up the channels for that window.
                    // Stride is in_channels*out_channels*sizeof(elem).
                    for (uint32_t block_weight_h = 0; block_weight_h < window_size_hw; block_weight_h++) {
                        uint32_t weight_row_start_tile_id = weight_current_block_start_tile_id;

                        // read the channels in one block.
                        for (uint32_t weight_tile_h_i = 0; weight_tile_h_i < core_in_channels_ntiles;
                             ++weight_tile_h_i) {
                            uint32_t weight_tile_id = weight_row_start_tile_id;

                            // loop over output channels, width of the output/weights.
                            for (uint32_t weight_tile_w_i = 0; weight_tile_w_i < weight_block_width_ntiles;
                                 ++weight_tile_w_i) {
                                s_weight.noc_async_read_tile(weight_tile_id, weight_write_l1_addr);
                                weight_write_l1_addr += weight_tile_nbytes;
                                weights_block_size_bytes += weight_tile_nbytes;
                                weight_tile_id += 1;
                            }  // for weight_block_w
                            weight_row_start_tile_id += weight_matrix_width_ntiles;
                        }  // for weight_block_h
                        weight_current_block_start_tile_id += weight_next_channel_stride_h;
                    }
                    noc_async_read_barrier();
                }
                cb_push_back(cb_id_weight, weight_block_num_tiles);
                weight_block_start_tile_id += weight_next_block_other_core_stride_h;
            }
            weight_start_tile_id += weight_next_block_this_core_stride_h;
            if (to_load_bias) {
#ifdef FUSE_BIAS
                cb_reserve_back(bias_cb_id, weight_block_width_ntiles);
                uint32_t bias_l1_addr = get_write_ptr(bias_cb_id);
                for (uint32_t weight_tile_w_i = 0; weight_tile_w_i < weight_block_width_ntiles; ++weight_tile_w_i) {
                    s_bias.noc_async_read_tile(bias_start_tile_id, bias_l1_addr);
                    bias_l1_addr += bias_pagesize;
                    bias_start_tile_id += 1;
                }
                noc_async_read_barrier();
                cb_push_back(bias_cb_id, weight_block_width_ntiles);
#endif
                to_load_bias = false;
            }
        }
    }
}
