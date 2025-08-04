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
    constexpr uint32_t ring_size = get_compile_time_arg_val(3);
    std::array<uint32_t, 4> fused_op_receiver_signal_semaphore_addr = {
        get_semaphore(get_compile_time_arg_val(4)),
        get_semaphore(get_compile_time_arg_val(5)),
        get_semaphore(get_compile_time_arg_val(6)),
        get_semaphore(get_compile_time_arg_val(7)),
    };
    DPRINT << "sem_wait_val: " << sem_wait_val << ENDL();
    DPRINT << "inter_cb_index: " << inter_cb_index << ENDL();
    DPRINT << "tensor0_page_size: " << tensor0_page_size << ENDL();
    DPRINT << "ring_size: " << ring_size << ENDL();
    DPRINT << "fused_op_receiver_signal_semaphore_addr[0]: " << fused_op_receiver_signal_semaphore_addr[0] << ENDL();
    DPRINT << "fused_op_receiver_signal_semaphore_addr[1]: " << fused_op_receiver_signal_semaphore_addr[1] << ENDL();
    DPRINT << "fused_op_receiver_signal_semaphore_addr[2]: " << fused_op_receiver_signal_semaphore_addr[2] << ENDL();
    DPRINT << "fused_op_receiver_signal_semaphore_addr[3]: " << fused_op_receiver_signal_semaphore_addr[3] << ENDL();
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

    const uint32_t mm_core_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t next_core_id_to_left = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t next_core_id_to_right = get_arg_val<uint32_t>(arg_idx++);

    DPRINT << "signal_semaphore_addr: " << signal_semaphore_addr << ENDL();
    DPRINT << "core_id: " << core_id << ENDL();
    DPRINT << "ring_index: " << ring_index << ENDL();
    DPRINT << "aggregated_tensor_addr: " << aggregated_tensor_addr << ENDL();
    DPRINT << "bbox_start_x: " << bbox_start_x << ENDL();
    DPRINT << "bbox_start_y: " << bbox_start_y << ENDL();
    DPRINT << "bbox_end_x: " << bbox_end_x << ENDL();
    DPRINT << "bbox_end_y: " << bbox_end_y << ENDL();
    DPRINT << "bbox_size: " << bbox_size << ENDL();
    DPRINT << "intermediate_tensor_shard_num_pages: " << intermediate_tensor_shard_num_pages << ENDL();
    DPRINT << "next_core_id_to_left: " << next_core_id_to_left << ENDL();
    DPRINT << "next_core_id_to_right: " << next_core_id_to_right << ENDL();

    DPRINT << "noc index: " << static_cast<uint32_t>(noc_index) << ENDL();

    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);

    // Set up for mcasting to mm workers
    volatile tt_l1_ptr uint32_t* fused_op_receiver_signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fused_op_receiver_signal_semaphore_addr[core_id]);
    noc_semaphore_set(fused_op_receiver_signal_semaphore_addr_ptr, VALID);

    volatile tt_l1_ptr uint32_t* fused_op_receiver_signal_semaphore_addr_ptr_next_core_right =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fused_op_receiver_signal_semaphore_addr[next_core_id_to_right]);

    DPRINT << "core_id: " << core_id << ENDL();

    // if (core_id != ring_index) {
    //     return;
    // }

    DPRINT << "core that handles local noc multicast: " << core_id << ENDL();

    // 1. Wait for global signal
    {
        DeviceZoneScopedN("data waiting");
        // uint64_t t1 = ckernel::read_wall_clock();
        noc_semaphore_wait_min(signal_semaphore_addr_ptr, sem_wait_val);
        noc_semaphore_set(signal_semaphore_addr_ptr, 0);
        // uint64_t t2 = ckernel::read_wall_clock();
        // DPRINT << "time taken(in us): " << (t2 - t1) << ENDL();
    }

    // 2. multicast data to mm cores
    // 2.1. Wait for local signal, if it's not the first core
    // volatile tt_l1_ptr uint32_t* recev_semaphore_addr_ptr =
    //     reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recev_semaphore_addr);
    if (core_id != ring_index) {  // don't need to wait if it's the first core
        noc_semaphore_wait_min(fused_op_receiver_signal_semaphore_addr_ptr_next_core_right, 1);
        noc_semaphore_set(fused_op_receiver_signal_semaphore_addr_ptr_next_core_right, 0);
    }

    size_t l1_read_addr = get_read_ptr(inter_cb_index);
    const uint64_t multicast_addr_noc = get_noc_multicast_addr(bbox_start_x, bbox_start_y, bbox_end_x, bbox_end_y, 0);
    uint64_t aggregated_tensor_addr_this_core =
        (uint64_t)aggregated_tensor_addr + mm_core_offset * intermediate_tensor_shard_num_pages * tensor0_page_size;
    const uint64_t multicast_addr = multicast_addr_noc | aggregated_tensor_addr_this_core;
    // noc_async_write_multicast(
    //     l1_read_addr, multicast_addr, intermediate_tensor_shard_num_pages * tensor0_page_size, bbox_size, true);

    noc_async_write_multicast_loopback_src(
        l1_read_addr, multicast_addr, intermediate_tensor_shard_num_pages * tensor0_page_size, bbox_size, true);
    // noc_async_write_multicast(l1_read_addr, multicast_addr, 0, bbox_size - 1, false);

    uint64_t multicast_sema_addr = multicast_addr_noc | (uint64_t)fused_op_receiver_signal_semaphore_addr[core_id];
    noc_semaphore_set_multicast_loopback_src(
        fused_op_receiver_signal_semaphore_addr[core_id], multicast_sema_addr, bbox_size, false);
    noc_async_write_barrier();
}
