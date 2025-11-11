// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // Runtime args
    uint32_t rt_args_idx = 0;
    const uint32_t out_num_nonzero_subblocks_h = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_num_nonzero_subblocks_w = get_arg_val<uint32_t>(rt_args_idx++);

    // Compile time args
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(3);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(4);

    // CB operations
    constexpr uint32_t cb_id_in1 = 1;
    cb_reserve_back(cb_id_in1, in1_block_num_tiles * num_blocks_inner_dim);
    cb_push_back(cb_id_in1, in1_block_num_tiles * num_blocks_inner_dim);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_4;
    cb_wait_front(
        cb_id_out0,
        batch * out_num_nonzero_subblocks_h * out_num_nonzero_subblocks_w * out_subblock_w * out_subblock_h);
}
