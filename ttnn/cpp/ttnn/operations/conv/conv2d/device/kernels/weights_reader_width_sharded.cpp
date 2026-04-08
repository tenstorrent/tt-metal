// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <api/dataflow/dataflow_api.h>
#include "api/debug/dprint.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

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
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(12);
    constexpr bool fuse_bias = get_compile_time_arg_val(13);
    constexpr auto s_weight_args = TensorAccessorArgs<14>();
    constexpr auto s_bias_args = TensorAccessorArgs<s_weight_args.next_compile_time_args_offset()>();

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
    const auto s_weight = TensorAccessor(s_weight_args, weight_addr_dram_base, weight_tile_nbytes);
    const uint32_t bias_pagesize =
        fuse_bias ? get_tile_size(bias_cb_id) : 0;  // dummy but valid value in case bias is not enabled
    const auto s_bias = TensorAccessor(s_bias_args, bias_addr_dram_base, bias_pagesize);

    experimental::CB weight_cb(cb_id_weight);
    experimental::CB bias_cb(bias_cb_id);
    experimental::Noc noc;

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
                weight_cb.reserve_back(weight_block_num_tiles);
                if (is_active) {
                    uint32_t weight_current_block_start_tile_id = weight_block_start_tile_id;

                    // for window size, picking up the channels for that window.
                    // Stride is in_channels*out_channels*sizeof(elem).
                    uint32_t weight_write_offset = 0;
                    for (uint32_t block_weight_h = 0; block_weight_h < window_size_hw; block_weight_h++) {
                        uint32_t weight_row_start_tile_id = weight_current_block_start_tile_id;

                        // read the channels in one block.
                        for (uint32_t weight_tile_h_i = 0; weight_tile_h_i < core_in_channels_ntiles;
                             ++weight_tile_h_i) {
                            uint32_t weight_tile_id = weight_row_start_tile_id;

                            // loop over output channels, width of the output/weights.
                            for (uint32_t weight_tile_w_i = 0; weight_tile_w_i < weight_block_width_ntiles;
                                 ++weight_tile_w_i) {
                                noc.async_read(
                                    s_weight,
                                    weight_cb,
                                    weight_tile_nbytes,
                                    {.page_id = weight_tile_id},
                                    {.offset_bytes = weight_write_offset});
                                weight_write_offset += weight_tile_nbytes;
                                weight_tile_id += 1;
                            }  // for weight_block_w
                            weight_row_start_tile_id += weight_matrix_width_ntiles;
                        }  // for weight_block_h
                        weight_current_block_start_tile_id += weight_next_channel_stride_h;
                    }
                    noc.async_read_barrier();
                }
                weight_cb.push_back(weight_block_num_tiles);
                weight_block_start_tile_id += weight_next_block_other_core_stride_h;
            }
            weight_start_tile_id += weight_next_block_this_core_stride_h;
            if (to_load_bias) {
                if constexpr (fuse_bias) {
                    bias_cb.reserve_back(weight_block_width_ntiles);
                    uint32_t bias_write_offset = 0;
                    for (uint32_t weight_tile_w_i = 0; weight_tile_w_i < weight_block_width_ntiles; ++weight_tile_w_i) {
                        noc.async_read(
                            s_bias,
                            bias_cb,
                            bias_pagesize,
                            {.page_id = bias_start_tile_id},
                            {.offset_bytes = bias_write_offset});
                        bias_write_offset += bias_pagesize;
                        bias_start_tile_id += 1;
                    }
                    noc.async_read_barrier();
                    bias_cb.push_back(weight_block_width_ntiles);
                }
                to_load_bias = false;
            }
        }
    }
}
