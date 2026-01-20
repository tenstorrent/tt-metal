// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute reader kernel for non-data compute cores
// Reads tile subset from intermediate and input tensors via NOC
// and pushes to local CBs for compute

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(0);       // remote data (intermediate tensor)
    constexpr uint32_t cb_in2 = get_compile_time_arg_val(1);       // local data (input tensor)
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);    // tiles this core processes
    constexpr uint32_t tile_offset = get_compile_time_arg_val(3);  // starting tile index
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t data_noc_x = get_compile_time_arg_val(5);
    constexpr uint32_t data_noc_y = get_compile_time_arg_val(6);

    size_t arg_idx = 0;
    const uint32_t input_tensor_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t intermediate_tensor_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t semaphore_addr = get_arg_val<uint32_t>(arg_idx++);

    // Wait locally for semaphore signal from receiver_reader (via mcast)
    volatile tt_l1_ptr uint32_t* local_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
    noc_semaphore_wait(local_sem_ptr, 1);
    noc_semaphore_set(local_sem_ptr, 0);

    // Calculate byte offset for this core's tile range
    const uint32_t byte_offset = tile_offset * page_size_bytes;
    const uint32_t total_bytes = num_tiles * page_size_bytes;

    // Read intermediate tensor data (remote data received via fabric, stored on data_core)
    cb_reserve_back(cb_in1, num_tiles);
    uint32_t l1_write_addr_in1 = get_write_ptr(cb_in1);
    uint64_t intermediate_noc_addr = get_noc_addr(data_noc_x, data_noc_y, intermediate_tensor_addr + byte_offset);
    noc_async_read(intermediate_noc_addr, l1_write_addr_in1, total_bytes);

    // Read input tensor data (local data, stored on data_core)
    cb_reserve_back(cb_in2, num_tiles);
    uint32_t l1_write_addr_in2 = get_write_ptr(cb_in2);
    uint64_t input_noc_addr = get_noc_addr(data_noc_x, data_noc_y, input_tensor_addr + byte_offset);
    noc_async_read(input_noc_addr, l1_write_addr_in2, total_bytes);

    noc_async_read_barrier();
    cb_push_back(cb_in1, num_tiles);
    cb_push_back(cb_in2, num_tiles);
}
