// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "ckernel.h"

void kernel_main() {
    DPRINT << "Kernel = worker_receiver" << ENDL();
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t sem_wait_val = get_compile_time_arg_val(0);
    constexpr uint32_t inter_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(2);
    DPRINT << "sem_wait_val: " << sem_wait_val << ENDL();
    DPRINT << "inter_cb_index: " << inter_cb_index << ENDL();
    DPRINT << "tensor0_page_size: " << tensor0_page_size << ENDL();

    // runtime args
    size_t arg_idx = 0;
    const uint32_t signal_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_id = get_arg_val<uint32_t>(
        arg_idx++);  // core id, corresponds to the id of which device it expect data from, will be reset later
    const uint32_t ring_index = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t aggregated_tensor_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t bbox_start_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t bbox_start_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t bbox_end_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t bbox_end_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t bbox_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t intermediate_tensor_shard_num_pages = get_arg_val<uint32_t>(arg_idx++);
    DPRINT << "signal_semaphore_addr: " << signal_semaphore_addr << ENDL();
    DPRINT << "core_id: " << core_id << ENDL();
    DPRINT << "ring_index: " << ring_index << ENDL();

    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);
    if (core_id != ring_index) {
        return;
    }

    // 1. Wait for signal
    {
        DeviceZoneScopedN("data waiting");
        // uint64_t t1 = ckernel::read_wall_clock();
        noc_semaphore_wait_min(signal_semaphore_addr_ptr, sem_wait_val);
        noc_semaphore_set(signal_semaphore_addr_ptr, 0);
        // uint64_t t2 = ckernel::read_wall_clock();
        // DPRINT << "time taken(in us): " << (t2 - t1) << ENDL();
    }

    // 2. multicast data to mm cores
    size_t l1_read_addr = get_read_ptr(inter_cb_index);
    const uint64_t multicast_addr =
        get_noc_multicast_addr(bbox_start_x, bbox_start_y, bbox_start_x + 1, bbox_start_y + 1, aggregated_tensor_addr);
    // noc_async_write_multicast(
    //     l1_read_addr, multicast_addr, intermediate_tensor_shard_num_pages * tensor0_page_size, bbox_size, true);
    noc_async_write_multicast(
        l1_read_addr, multicast_addr, intermediate_tensor_shard_num_pages * tensor0_page_size, 4, false);
    noc_async_write_barrier();
}
