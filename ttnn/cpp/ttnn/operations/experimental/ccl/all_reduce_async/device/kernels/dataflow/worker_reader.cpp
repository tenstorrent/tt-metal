// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include <cstdint>
#include <utility>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t cb0_id = get_compile_time_arg_val(1);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(2);
// reduction compile time args
constexpr uint32_t cb_id = get_compile_time_arg_val(3);
constexpr uint32_t total_num_reduction_tiles = get_compile_time_arg_val(4);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    const uint32_t is_worker = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t is_reducer = get_arg_val<uint32_t>(arg_idx++);
    if (is_worker == 0 && is_reducer == 0) {
        return;
    }
    if (is_worker == 1) {
        address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
        uint32_t num_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
        uint32_t num_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
        uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
        uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
        tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
        arg_idx += num_cores;
        tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
        arg_idx += num_cores;

        // interleaved addrgen
        uint32_t tiles_read = 0;
        uint32_t shard_tile_id = first_core_tile_start_offset;
        uint32_t core_id = 0;
        while (tiles_read < num_tiles_to_read) {
            uint32_t num_tiles_to_read_this_core =
                std::min(num_tiles_per_core - shard_tile_id, num_tiles_to_read - tiles_read);
            cb_reserve_back(cb0_id, num_tiles_to_read_this_core);
            const uint32_t l1_write_addr = get_write_ptr(cb0_id);
            uint64_t read_addr = get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], tensor_address0);
            read_addr += shard_tile_id * tensor0_page_size;

            noc_async_read(read_addr, l1_write_addr, num_tiles_to_read_this_core * tensor0_page_size);
            noc_async_read_barrier();

            cb_push_back(cb0_id, num_tiles_to_read_this_core);
            tiles_read += num_tiles_to_read_this_core;
            shard_tile_id = 0;
            core_id++;
        }
    }
    if (is_reducer == 1) {
        const uint32_t sem_id = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t signal_semaphore_addr = get_semaphore(sem_id);
        volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);

        // 1. Wait for signal from All-Gather worker
        noc_semaphore_wait(signal_semaphore_addr_ptr, VALID);
        if (is_reducer == 1 && is_worker == 0) {
            noc_semaphore_set(signal_semaphore_addr_ptr, 0);
        }
        // 2. Signal compute kernel to start processing
        cb_push_back(cb_id, total_num_reduction_tiles);
    }
}
