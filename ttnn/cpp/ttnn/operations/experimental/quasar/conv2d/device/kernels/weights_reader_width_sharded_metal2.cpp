// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of weights_reader_width_sharded.cpp (width-sharded conv2d).
// Algorithm body is identical to the legacy kernel; only the host-binding surface is migrated:
//   - CB-index CTAs (cb_id_weight / bias_cb_id) -> dfb::weights / dfb::bias
//   - weight/bias TensorAccessorArgs + base-address RTAs -> tensor::weights / tensor::bias bindings
//   - remaining positional CTAs -> get_arg(args::name)
//   - remaining RTAs -> get_arg(args::name)
//   - experimental::CB -> DataflowBuffer; get_tile_size(cb) -> cb.get_entry_size()

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
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
    // Bias presence is carried as the FUSE_BIAS preprocessor define (gates the conditionally-bound
    // dfb::bias / tensor::bias references at the preprocessor level).

    const uint32_t init_weight_start_tile_id = get_arg(args::init_weight_start_tile_id);
    const uint32_t is_active = get_arg(args::is_active);

    DataflowBuffer weight_cb(dfb::weights);
    Noc noc;

    const uint32_t weight_tile_nbytes = weight_cb.get_entry_size();
    const auto s_weight = TensorAccessor(tensor::weights);

#ifdef FUSE_BIAS
    DataflowBuffer bias_cb(dfb::bias);
    const uint32_t bias_pagesize = bias_cb.get_entry_size();
    const auto s_bias = TensorAccessor(tensor::bias);
#endif

    bool to_load_bias = true;

    for (uint32_t act_block_h_index = 0; act_block_h_index < act_num_blocks_h; act_block_h_index++) {
        uint32_t weight_start_tile_id = init_weight_start_tile_id;
        uint32_t bias_start_tile_id = init_weight_start_tile_id;

        for (uint32_t local_weight_block_index = 0; local_weight_block_index < local_weight_height_blocks;
             local_weight_block_index++) {
            uint32_t weight_block_start_tile_id = weight_start_tile_id;

            for (uint32_t remote_weight_block_index = 0; remote_weight_block_index < remote_weight_height_blocks;
                 remote_weight_block_index++) {
                weight_cb.reserve_back(weight_block_num_tiles);
                if (is_active) {
                    uint32_t weight_current_block_start_tile_id = weight_block_start_tile_id;

                    uint32_t weight_write_offset = 0;
                    for (uint32_t block_weight_h = 0; block_weight_h < window_size_hw; block_weight_h++) {
                        uint32_t weight_row_start_tile_id = weight_current_block_start_tile_id;

                        for (uint32_t weight_tile_h_i = 0; weight_tile_h_i < core_in_channels_ntiles;
                             ++weight_tile_h_i) {
                            uint32_t weight_tile_id = weight_row_start_tile_id;

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
#endif
                to_load_bias = false;
            }
        }
    }
}
