// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

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

    // DPRINT << "multicast_chunk_size_in_tiles: " << multicast_chunk_size_in_tiles << ENDL();
    // DPRINT << "multicast_chunk_width_in_tiles: " << multicast_chunk_width_in_tiles << ENDL();
    // DPRINT << "shard_height_in_tiles: " << shard_height_in_tiles << ENDL();
    // DPRINT << "chunk_indices[0]: " << chunk_indices[0] << ENDL();
    // DPRINT << "chunk_indices[1]: " << chunk_indices[1] << ENDL();
    // DPRINT << "chunk_indices[2]: " << chunk_indices[2] << ENDL();
    // DPRINT << "chunk_indices[3]: " << chunk_indices[3] << ENDL();

    cb_reserve_back(cb_id_in0, multicast_chunk_size_in_tiles * num_multicast_steps);
    for (uint32_t istep = 0; istep < num_multicast_steps; istep++) {
        // DPRINT << "MM RD waiting for local sem ..." << ENDL();
        volatile tt_l1_ptr uint32_t* fused_op_receiver_signal_semaphore_addr_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                fused_op_receiver_signal_semaphore_addr[chunk_indices[istep]]);
        {
            // DeviceZoneScopedN("data waiting in0");
            noc_semaphore_wait_min(fused_op_receiver_signal_semaphore_addr_ptr, 1);
            noc_semaphore_set(fused_op_receiver_signal_semaphore_addr_ptr, 0);
        }
        // DPRINT << "MM RD done waiting for local sem ..." << ENDL();

        {
            // DeviceZoneScopedN("data pushing in0");
            cb_push_back(cb_id_in0, multicast_chunk_size_in_tiles);  // put zone scopes here
        }
    }
}
