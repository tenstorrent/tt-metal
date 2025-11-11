// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

void kernel_main() {
    // READER
    uint32_t rt_args_idx = 0;
    // in1 tensor args
    const uint32_t in1_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t in1_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);
    // in1 mcast args
    const uint32_t in1_mcast_dest_noc_start_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in1_mcast_dest_noc_start_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in1_mcast_dest_noc_end_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in1_mcast_dest_noc_end_y = get_arg_val<uint32_t>(rt_args_idx++);

    // sparsity args
    const uint32_t sparsity_addr = get_arg_val<uint32_t>(rt_args_idx++);

    // WRITER
    // out tensor args
    const uint32_t out_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);

    // padding args (READER)
    const uint32_t last_block_w = get_arg_val<uint32_t>(rt_args_idx++);
    // padding args (WRITER)
    const uint32_t out_num_nonzero_subblocks_h = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_subblock_h = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t padded_block_tiles_h_skip = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_num_nonzero_subblocks_w = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_num_nonzero_subblocks_w = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t out_last_subblock_w = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t padded_subblock_tiles_addr_skip = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t padded_block_tiles_w_skip = get_arg_val<uint32_t>(rt_args_idx++);

    // COMPILE TIME ARGS
    // READER
    // in1 tensor args
    constexpr uint32_t in1_tensor_stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t in1_tensor_stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t in1_tensor_next_block_stride = get_compile_time_arg_val(2);
    constexpr uint32_t in1_tensor_next_w_dim_block_stride = get_compile_time_arg_val(3);
    // in1 block args
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(4);
    constexpr uint32_t in1_block_h = get_compile_time_arg_val(5);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(6);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(7);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(8);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(9);

    // in1 mcast args
    uint32_t in1_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(10));
    uint32_t in1_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(11));
    constexpr uint32_t in1_mcast_num_dests = get_compile_time_arg_val(12);
    constexpr uint32_t in1_mcast_num_cores = get_compile_time_arg_val(13);
    // batch args
    constexpr uint32_t KtNt = get_compile_time_arg_val(14);
    constexpr uint32_t batch = get_compile_time_arg_val(15);
    constexpr uint32_t bcast_B = get_compile_time_arg_val(16);
    // sparsity args
    constexpr uint32_t batchB = get_compile_time_arg_val(17);
    constexpr uint32_t sparsity_pagesize = get_compile_time_arg_val(18);

    // WRITER
    // out tensor args
    constexpr uint32_t out_tensor_stride_w = get_compile_time_arg_val(19);
    constexpr uint32_t out_tensor_stride_h = get_compile_time_arg_val(20);
    constexpr uint32_t out_tensor_next_subblock_stride_w = get_compile_time_arg_val(21);
    constexpr uint32_t out_tensor_next_subblock_stride_h = get_compile_time_arg_val(22);
    constexpr uint32_t out_tensor_next_w_dim_block_stride = get_compile_time_arg_val(23);
    constexpr uint32_t out_tensor_next_h_dim_block_stride = get_compile_time_arg_val(24);
    // out subblock args
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(25);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(26);
    constexpr uint32_t out_subblock_tile_count = get_compile_time_arg_val(27);
    // batch args
    constexpr uint32_t MtNt = get_compile_time_arg_val(28);  // if 0
    // Don't need batch; same as batch from READER args

    // When sparsity is disabled, we just loop once
    constexpr uint32_t batchB_lim = batchB == 0 ? 1u : batchB;

    constexpr uint32_t one_tile = 1;

    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    constexpr const uint32_t in1_tile_hw = get_tile_hw(cb_id_in1);
    constexpr uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size_bytes;

    cb_reserve_back(cb_id_in1, in1_block_num_tiles * num_blocks_inner_dim);
    cb_push_back(cb_id_in1, in1_block_num_tiles * num_blocks_inner_dim);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_4;
    constexpr uint32_t output_single_tile_size_bytes = get_tile_size(cb_id_out0);
    constexpr const uint32_t output_tile_hw = get_tile_hw(cb_id_out0);
    const auto s = TensorAccessor(out_args, out_tensor_addr, output_single_tile_size_bytes);

    // sparsity accessor
    constexpr uint32_t cb_id_sparsity = tt::CBIndex::c_7;
    const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr, sparsity_pagesize);

    cb_wait_front(
        cb_id_out0,
        batch * out_num_nonzero_subblocks_h * out_num_nonzero_subblocks_w * out_subblock_w * out_subblock_h);
}
