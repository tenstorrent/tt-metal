// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
<<<<<<< HEAD:ttnn/cpp/ttnn/experimental/tt_dnn/op_library/all_gather/kernels/dataflow/worker_sharded_ring_gather_send_reader.cpp
#include "ttnn/cpp/ttnn/experimental/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"
=======
#include "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"
>>>>>>> 9c82037fb9... #9486: Move CCL kernel files to TTNN:ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_sharded_ring_gather_send_reader.cpp

void kernel_main() {
    constexpr ShardType shard_type = static_cast<ShardType>(get_compile_time_arg_val(0));
    constexpr uint32_t num_transfers = get_compile_time_arg_val(1);

    ShardAddrGen<shard_type> input_tensor_shard_reader;
    ShardAddrGen<shard_type> output_tensor_shard_reader;

    uint32_t arg_index = 0;
    volatile tt_l1_ptr uint32_t* local_semaphore_address = get_arg_val<volatile tt_l1_ptr uint32_t*>(arg_index++);
    uint32_t const num_shards_per_transfer = get_arg_val<uint32_t>(arg_index++);
    uint32_t const shards_per_eth_l1_buffer = get_arg_val<uint32_t>(arg_index++);
    uint32_t const half_cb_n_pages = get_arg_val<uint32_t>(arg_index++);
    ShardAddrGen<shard_type>::build_with_placement_new(&input_tensor_shard_reader, arg_index);
    arg_index += input_tensor_shard_reader.get_num_args_consumed();
    ShardAddrGen<shard_type>::build_with_placement_new(&output_tensor_shard_reader, arg_index);
    arg_index += output_tensor_shard_reader.get_num_args_consumed();

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    for (uint32_t c = 0; c < num_shards_per_transfer; c += shards_per_eth_l1_buffer) {
        uint32_t num_shards_to_send = std::min(shards_per_eth_l1_buffer, num_shards_per_transfer - c);
        read_shard_from_input_tensor_sharded(cb_id_in0, input_tensor_shard_reader, num_shards_to_send);
        ASSERT(half_cb_n_pages >= num_shards_to_send);
        if (half_cb_n_pages > num_shards_to_send) {
            push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - num_shards_to_send);
        }
    }

    uint32_t sem_idx = 1;

    for (uint32_t i = 1; i < num_transfers; ++i) {

        for (uint32_t c = 0; c < num_shards_per_transfer; c += shards_per_eth_l1_buffer) {
            uint32_t num_shards_to_send = std::min(shards_per_eth_l1_buffer, num_shards_per_transfer - c);
            noc_semaphore_wait_min(local_semaphore_address, sem_idx);
            sem_idx += num_shards_to_send;
            read_chunk_from_output_tensor_sharded(cb_id_in0, output_tensor_shard_reader, num_shards_to_send);  // 1 chunk == 1 shard for now
            if (half_cb_n_pages > num_shards_to_send) {
                push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - num_shards_to_send);
            }
        }
    }

}
