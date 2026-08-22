// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Width-sharded Conv2D weights reader using the Metalium 2.0 kernel-binding surface:
//   - CB-index CTAs (cb_id_weight / bias_cb_id) -> dfb::weights / dfb::bias
//   - weight/bias TensorAccessorArgs + base-address RTAs -> tensor::weights / tensor::bias bindings
//   - compile-time choices -> TT_KERNEL template arguments
//   - RTAs -> TT_KERNEL function arguments
//   - experimental::CB -> DataflowBuffer; get_tile_size(cb) -> cb.get_entry_size()

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

template <
    uint32_t core_in_channels_ntiles,
    uint32_t window_size_hw,
    uint32_t weight_block_width_ntiles,
    uint32_t weight_block_num_tiles,
    uint32_t weight_matrix_width_ntiles,
    uint32_t weight_next_channel_stride_h,
    uint32_t weight_next_block_this_core_stride_h,
    uint32_t weight_next_block_other_core_stride_h,
    uint32_t remote_weight_height_blocks,
    uint32_t local_weight_height_blocks,
    uint32_t act_num_blocks_h,
    uint32_t fuse_bias>
TT_KERNEL void weights_reader_width_sharded(uint32_t init_weight_start_tile_id, uint32_t is_active) {
    DataflowBuffer weight_cb(dfb::weights);
    Noc noc;

    const uint32_t weight_tile_nbytes = weight_cb.get_entry_size();
    const auto s_weight = TensorAccessor(tensor::weights);

    DataflowBuffer bias_cb(dfb::bias);
    const uint32_t bias_pagesize = fuse_bias ? bias_cb.get_entry_size() : 0;
    const auto s_bias = TensorAccessor(tensor::bias);

    bool to_load_bias = true;

    for (uint32_t act_block_h_index = 0; act_block_h_index < act_num_blocks_h; act_block_h_index++) {
        uint32_t weight_start_tile_id = init_weight_start_tile_id;
        uint32_t bias_start_tile_id = init_weight_start_tile_id;

        // Interleave each core-local activation-width block with compute. The outer stride advances by
        // act_block_w * output_channels; the remote-core stride advances by the per-core input-channel
        // slice; and each window position advances by the full input-channel matrix height.
        for (uint32_t local_weight_block_index = 0; local_weight_block_index < local_weight_height_blocks;
             local_weight_block_index++) {
            uint32_t weight_block_start_tile_id = weight_start_tile_id;

            for (uint32_t remote_weight_block_index = 0; remote_weight_block_index < remote_weight_height_blocks;
                 remote_weight_block_index++) {
                weight_cb.reserve_back(weight_block_num_tiles);
                if (is_active) {
                    uint32_t weight_current_block_start_tile_id = weight_block_start_tile_id;

                    // Gather one convolution window, then all input-channel tiles for that position,
                    // and finally the output-channel tiles forming the contiguous destination block.
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
