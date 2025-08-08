// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "debug/dprint.h"

enum class CORE_TYPE : uint8_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };
void kernel_main() {
    std::array<uint32_t, 4> fused_op_receiver_signal_semaphore_addr = {
        get_semaphore(get_compile_time_arg_val(7)),
        get_semaphore(get_compile_time_arg_val(8)),
        get_semaphore(get_compile_time_arg_val(9)),
        get_semaphore(get_compile_time_arg_val(10)),
    };
    std::array<uint32_t, 4> chunk_indices = {
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(14),
    };

    // Compile time args
    constexpr uint32_t multicast_chunk_width_in_tiles = get_compile_time_arg_val(0);  // 1/4 of K per step
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(1);           // per_core_M
    constexpr uint32_t batch = get_compile_time_arg_val(2);

    // Multicast specific
    constexpr uint32_t num_multicast_steps = get_compile_time_arg_val(3);  // Always 4
    uint32_t signal_semaphore_addr = get_semaphore(get_compile_time_arg_val(4));

    // Runtime args
    uint32_t rt_args_idx = 0;
    uint32_t core_type = get_arg_val<uint32_t>(rt_args_idx++);

    uint32_t ring_idx = get_arg_val<uint32_t>(rt_args_idx++);         // Core index for multicast addressing
    uint32_t multicast_steps = get_arg_val<uint32_t>(rt_args_idx++);  // Should be 4
    if (core_type == (uint32_t)CORE_TYPE::IDLE_CORE) {
        return;
    }

    // No ring topology arguments needed for multicast
    // No unpadded widths array needed since uniform 1/4 chunks

    volatile tt_l1_ptr uint32_t* l1_signal_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);
    // No remote signal semaphore needed for multicast approach

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_id_in2 = get_compile_time_arg_val(6);

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t multicast_chunk_size_in_tiles = multicast_chunk_width_in_tiles * shard_height_in_tiles;
    constexpr uint32_t multicast_chunk_size_bytes = multicast_chunk_size_in_tiles * in0_single_tile_size_bytes;

    cb_reserve_back(cb_id_in0, multicast_chunk_size_in_tiles * num_multicast_steps);
    for (uint32_t istep = 0; istep < num_multicast_steps; istep++) {
        volatile tt_l1_ptr uint32_t* fused_op_receiver_signal_semaphore_addr_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                fused_op_receiver_signal_semaphore_addr[chunk_indices[istep]]);

        noc_semaphore_wait_min(fused_op_receiver_signal_semaphore_addr_ptr, 1);
        noc_semaphore_set(fused_op_receiver_signal_semaphore_addr_ptr, 0);
        cb_push_back(cb_id_in0, multicast_chunk_size_in_tiles);
    }

    /*
    // Reserving/pushing the local shard is done in compute
    cb_reserve_back(cb_id_in2, (ring_size - 1) * shard_size_in_tiles);

    uint32_t local_shard_read_addr = get_read_ptr(cb_id_in0);
    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in2);

    uint32_t hop_core_offset = static_cast<uint32_t>(is_hop_core);

    for (uint32_t shard_cnt = hop_core_offset; shard_cnt < ring_size; shard_cnt++) {
        uint32_t curr_ring_idx = (ring_idx + shard_cnt) % ring_size;
        bool skip_send = unpadded_in0_shard_widths_in_tiles[curr_ring_idx] == 0 && !is_hop_core;

        uint32_t curr_shard_write_addr = l1_write_addr_in0 + shard_size_bytes * (shard_cnt - hop_core_offset);
        uint64_t remote_curr_shard_write_addr =
            get_noc_addr(next_core_noc_x, next_core_noc_y, curr_shard_write_addr, noc);
        uint32_t curr_shard_read_addr =
            shard_cnt == 0 ? local_shard_read_addr : l1_write_addr_in0 + shard_size_bytes * (shard_cnt - 1);

        // Wait for signal from previous core that data has been added to this core's in0
        noc_semaphore_wait_min(l1_signal_sem_addr, shard_cnt);

        // Send data to next core
        if (shard_cnt < ring_size - 1 || is_hop_core) {  // Skip sending the last shard
            if (!skip_send) {
                noc_async_write(curr_shard_read_addr, remote_curr_shard_write_addr, shard_size_bytes, noc);
            }

            // Signal the next core that data is ready
            noc_semaphore_inc(remote_signal_semaphore_addr, 1, noc);
        }

        // Do stuff for matmul fusion here
        if (shard_cnt > 0) {
            cb_push_back(cb_id_in2, shard_size_in_tiles);
        }
    }


    noc_async_atomic_barrier();
    */
}
