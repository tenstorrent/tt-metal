// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <api/dataflow/dataflow_api.h>
#include "api/debug/dprint.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t cb_id_weight = dfb::weights;
    constexpr uint32_t core_in_channels_ntiles = get_arg(args::core_in_channels_ntiles);
    constexpr uint32_t window_size_hw = get_arg(args::window_size_hw);

    // weight_block_width_ntiles corresponds to the full output width of each core.
    constexpr uint32_t weight_block_width_ntiles = get_arg(args::weight_block_width_ntiles);
    constexpr uint32_t weight_block_num_tiles = get_arg(args::weight_block_num_tiles);
    constexpr uint32_t weight_matrix_width_ntiles = get_arg(args::weight_matrix_width_ntiles);
    constexpr uint32_t weight_next_channel_stride_h = get_arg(args::weight_next_channel_stride_h);
    constexpr uint32_t weight_next_block_this_core_stride_h = get_arg(args::weight_next_block_this_core_stride_h);
    constexpr uint32_t weight_next_block_other_core_stride_h = get_arg(args::weight_next_block_other_core_stride_h);
    constexpr uint32_t remote_weight_height_blocks = get_arg(args::remote_weight_height_blocks);
    constexpr uint32_t local_weight_height_blocks = get_arg(args::local_weight_height_blocks);
    constexpr uint32_t act_num_blocks_h = get_arg(args::act_num_blocks_h);
    // BIAS DFB bound only when has_bias; alias to weights when unbound so bias_cb
    // construction stays well-formed (all real uses are under `if constexpr (fuse_bias)`).
#ifdef FUSE_BIAS
    constexpr uint32_t bias_cb_id = dfb::bias;
#else
    constexpr uint32_t bias_cb_id = dfb::weights;
#endif
    constexpr bool fuse_bias = get_arg(args::fuse_bias);

    const uint32_t init_weight_start_tile_id = get_arg(args::init_weight_start_tile_id);
    const uint32_t is_active = get_arg(args::is_active);

    const uint32_t weight_tile_nbytes = get_tile_size(cb_id_weight);
    // Weight (and optional bias) base addresses now arrive via the typed tensor
    // binding channel; TensorAccessor(ta::name) packs the per-enqueue base address.
    const auto s_weight = TensorAccessor(ta::weights);
    const uint32_t bias_pagesize =
        fuse_bias ? get_tile_size(bias_cb_id) : 0;  // dummy but valid value in case bias is not enabled
#ifdef FUSE_BIAS
    const auto s_bias = TensorAccessor(ta::bias);
#endif

    experimental::CB weight_cb(cb_id_weight);
    experimental::CB bias_cb(bias_cb_id);
    Noc noc;

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
#ifdef FUSE_BIAS
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
#endif
                to_load_bias = false;
            }
        }
    }
}
