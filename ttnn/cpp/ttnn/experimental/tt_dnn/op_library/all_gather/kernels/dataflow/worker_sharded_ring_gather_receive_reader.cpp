// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "debug/assert.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {
    // TODO: Update the interleaver receive reader kernel invocation to just be able to use this
    constexpr ShardType shard_type = static_cast<ShardType>(get_compile_time_arg_val(0));
    ShardAddrGen<shard_type> input_tensor_shard_writer;

    // Info about the eth receiver eth core (producer of this core)
    // TODO: Make this arch agnostic

    uint32_t arg_index = 0;
    const uint32_t eth_receiver_noc_x = get_arg_val<uint32_t>(arg_index++);
    const uint32_t eth_receiver_noc_y = get_arg_val<uint32_t>(arg_index++);
    const uint32_t eth_receiver_l1_base_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t eth_receiver_l1_semaphore_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t receiver_read_sem_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t input_shards_per_eth_buffer = get_arg_val<uint32_t>(arg_index++);
    const uint32_t num_transfers = get_arg_val<uint32_t>(arg_index++);
    const uint32_t half_cb_n_pages = get_arg_val<uint32_t>(arg_index++);

    ShardAddrGen<shard_type>::build_with_placement_new(&input_tensor_shard_writer, arg_index);
    arg_index += input_tensor_shard_writer.get_num_args_consumed();
    ASSERT(eth_receiver_noc_x >= 1 && eth_receiver_noc_x < 12  && (eth_receiver_noc_y == 0 || eth_receiver_noc_y == 6));

    // Eth receiver will set this semaphore when data is available
    volatile tt_l1_ptr uint32_t* receiver_read_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_read_sem_addr);

    // Address of the buffer on the eth receiver, this is different per receiver worker core
    const uint64_t eth_receiver_l1_base_noc_addr = get_noc_addr(eth_receiver_noc_x, eth_receiver_noc_y, eth_receiver_l1_base_addr);

    // Address of the semaphore on the eth receiver, this is the same per receiver worker core
    const uint64_t eth_receiver_l1_semaphore_noc_addr = get_noc_addr(eth_receiver_noc_x, eth_receiver_noc_y, eth_receiver_l1_semaphore_addr);

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    uint32_t shards_per_ring_index = input_tensor_shard_writer.get_num_dest_cores();
    uint32_t const shard_size = input_tensor_shard_writer.get_shard_size_in_bytes();
    for (uint32_t t = 0; t < num_transfers; t++) {
        for (uint32_t i = 0; i < shards_per_ring_index; i += input_shards_per_eth_buffer) {
            uint32_t shards_to_send = std::min(input_shards_per_eth_buffer, shards_per_ring_index - i);
            noc_semaphore_wait(receiver_read_semaphore_addr_ptr, 1);
            noc_semaphore_set(receiver_read_semaphore_addr_ptr, 0);
            fetch_chunk_sharded(
                cb_id_in0,
                shards_to_send,
                shard_size,
                eth_receiver_l1_base_noc_addr);
            noc_semaphore_inc(eth_receiver_l1_semaphore_noc_addr, 1);
            if (half_cb_n_pages > shards_to_send) {
                push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - shards_to_send);
            }
        }
    }
}
