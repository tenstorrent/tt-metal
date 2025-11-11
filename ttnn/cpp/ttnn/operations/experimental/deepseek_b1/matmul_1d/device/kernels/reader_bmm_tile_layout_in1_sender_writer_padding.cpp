// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // Runtime args - skip unused args to maintain arg index alignment
    uint32_t rt_args_idx = 0;
    rt_args_idx += 10;  // Skip: in1_tensor_addr, in1_tensor_start_tile_id, mcast coords (4), sparsity_addr,
                        // out_tensor_addr, out_tensor_start_tile_id, last_block_w
    const uint32_t out_num_nonzero_subblocks_h = get_arg_val<uint32_t>(rt_args_idx++);
    rt_args_idx += 2;  // Skip: out_last_subblock_h, padded_block_tiles_h_skip
    const uint32_t out_num_nonzero_subblocks_w = get_arg_val<uint32_t>(rt_args_idx++);
    rt_args_idx += 4;  // Skip: out_last_num_nonzero_subblocks_w, out_last_subblock_w, padded_subblock_tiles_addr_skip,
                       // padded_block_tiles_w_skip

    // Compile time args - only keep what's used
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(7);
    constexpr uint32_t batch = get_compile_time_arg_val(15);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(25);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(26);

    // CB operations
    constexpr uint32_t cb_id_in1 = 1;
    cb_reserve_back(cb_id_in1, in1_block_num_tiles * num_blocks_inner_dim);
    cb_push_back(cb_id_in1, in1_block_num_tiles * num_blocks_inner_dim);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_4;
    cb_wait_front(
        cb_id_out0,
        batch * out_num_nonzero_subblocks_h * out_num_nonzero_subblocks_w * out_subblock_w * out_subblock_h);
}
