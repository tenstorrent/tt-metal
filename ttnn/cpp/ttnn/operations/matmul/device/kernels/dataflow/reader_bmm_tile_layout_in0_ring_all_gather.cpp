// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "api/debug/dprint.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc_semaphore.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"

enum class CORE_TYPE : uint8_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

void kernel_main() {
    // Compile time args
    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);

    // All Gather specific
    constexpr uint32_t ring_size = get_compile_time_arg_val(3);

    // Runtime args
    uint32_t rt_args_idx = 0;
    uint32_t core_type = get_arg_val<uint32_t>(rt_args_idx++);
    if (core_type == (uint32_t)CORE_TYPE::IDLE_CORE) {
        return;
    }
    bool is_hop_core = core_type == (uint32_t)CORE_TYPE::HOP_CORE;

    uint32_t ring_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t next_core_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t next_core_noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t noc_id = get_arg_val<uint32_t>(rt_args_idx++);
    bool end_of_hop = (bool)get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t* unpadded_in0_shard_widths_in_tiles = nullptr;
    if (!is_hop_core) {
        unpadded_in0_shard_widths_in_tiles = (uint32_t*)get_arg_addr(rt_args_idx);
        rt_args_idx += ring_size;
    }

    experimental::Noc noc_obj(noc_id);
    experimental::Semaphore<> signal_sem(get_compile_time_arg_val(4));

    constexpr uint32_t cb_id_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_id_in2 = get_named_compile_time_arg_val("cb_in2");

    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::CircularBuffer cb_in2(cb_id_in2);

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t shard_size_in_tiles = shard_width_in_tiles * shard_height_in_tiles;
    constexpr uint32_t shard_size_bytes = shard_size_in_tiles * in0_single_tile_size_bytes;

    // Reserving/pushing the local shard is done in compute
    cb_in2.reserve_back((ring_size - 1) * shard_size_in_tiles);

    uint32_t local_shard_read_addr = cb_in0.get_read_ptr();
    uint32_t l1_write_addr_in0 = cb_in2.get_write_ptr();

    uint32_t hop_core_offset = static_cast<uint32_t>(is_hop_core);

    for (uint32_t shard_cnt = hop_core_offset; shard_cnt < ring_size; shard_cnt++) {
        uint32_t curr_ring_idx = (ring_idx + shard_cnt) % ring_size;
        bool skip_send = !is_hop_core && unpadded_in0_shard_widths_in_tiles[curr_ring_idx] == 0;

        uint32_t curr_shard_write_addr = l1_write_addr_in0 + shard_size_bytes * (shard_cnt - hop_core_offset);
        uint32_t curr_shard_read_addr =
            shard_cnt == 0 ? local_shard_read_addr : l1_write_addr_in0 + shard_size_bytes * (shard_cnt - 1);

        // Wait for signal from previous core that data has been added to this core's in0
        signal_sem.wait_min(shard_cnt);

        // Send data to next core
        if (shard_cnt < ring_size - 1 || is_hop_core) {  // Skip sending the last shard
            if (!skip_send) {
                experimental::UnicastEndpoint dst_ep;
                noc_obj.async_write(
                    experimental::CoreLocalMem<uint32_t>(curr_shard_read_addr),
                    dst_ep,
                    shard_size_bytes,
                    {},
                    {.noc_x = next_core_noc_x, .noc_y = next_core_noc_y, .addr = curr_shard_write_addr});
            }

            // Signal the next core that data is ready
            signal_sem.up(noc_obj, next_core_noc_x, next_core_noc_y, 1);
        }

        // Do stuff for matmul fusion here
        if (shard_cnt > 0) {
            cb_in2.push_back(shard_size_in_tiles);
        }
    }

    if (!is_hop_core) {
        for (uint32_t b = 0; b < batch - 1; ++b) {  // for rest batches, not need to gather in0 anymore
            cb_in2.reserve_back((ring_size - 1) * shard_size_in_tiles);
            cb_in2.push_back((ring_size - 1) * shard_size_in_tiles);
        }
    }
    noc_obj.async_atomic_barrier();
}
