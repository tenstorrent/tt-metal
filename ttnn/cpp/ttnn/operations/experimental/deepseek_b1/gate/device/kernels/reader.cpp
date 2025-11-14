// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

void kernel_main() {
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in0_block_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(2);
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(3));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(4));
    constexpr uint32_t in0_mcast_num_dests = get_compile_time_arg_val(5);
    constexpr uint32_t in0_mcast_num_cores = get_compile_time_arg_val(6);
    constexpr uint32_t num_x = get_compile_time_arg_val(7);
    constexpr uint32_t num_y = get_compile_time_arg_val(8);
    constexpr bool transpose_mcast = (bool)get_compile_time_arg_val(9);
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(11);
    constexpr uint32_t batch = get_compile_time_arg_val(12);

    uint32_t rt_args_idx = 0;
    const uint32_t sender_id = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_start_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_end_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t in0_mcast_dest_noc_end_y = get_arg_val<uint32_t>(rt_args_idx++);
    tt_l1_ptr uint32_t* in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_x)));
    tt_l1_ptr uint32_t* in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(increment_arg_idx(rt_args_idx, num_y)));

    constexpr uint32_t cb_id_in0 = 0;

    cb_reserve_back(cb_id_in0, batch * in0_block_num_tiles);
    cb_push_back(cb_id_in0, in0_block_num_tiles);
    // noc_async_write_barrier();
}
