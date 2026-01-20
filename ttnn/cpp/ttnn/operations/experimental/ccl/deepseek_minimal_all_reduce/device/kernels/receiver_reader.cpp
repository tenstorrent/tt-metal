// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include <cstdint>
#include <utility>
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/api_common.h"

using namespace tt::tt_fabric::linear::experimental;
using namespace tt::tt_fabric::common::experimental;

using tt::data_movement::common::tt_memmove;

void kernel_main() {
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t receiver_cb = get_compile_time_arg_val(1);
    constexpr uint32_t alignment = get_compile_time_arg_val(2);
    constexpr uint32_t cb_compute = get_compile_time_arg_val(3);
    constexpr uint32_t input_num_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t packet_size_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t data_noc_x = get_compile_time_arg_val(7);
    constexpr uint32_t data_noc_y = get_compile_time_arg_val(8);
    constexpr uint32_t remote_sender_noc_x = get_compile_time_arg_val(9);
    constexpr uint32_t remote_sender_noc_y = get_compile_time_arg_val(10);
    constexpr uint32_t mcast_start_x = get_compile_time_arg_val(11);
    constexpr uint32_t mcast_start_y = get_compile_time_arg_val(12);
    constexpr uint32_t mcast_end_x = get_compile_time_arg_val(13);
    constexpr uint32_t mcast_end_y = get_compile_time_arg_val(14);
    constexpr uint32_t mcast_num_dests = get_compile_time_arg_val(15);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
    constexpr uint8_t sender_num_hops = 1;
    constexpr uint32_t num_connections = 1;

    size_t arg_idx = 0;
    uint32_t tensor_address0 = get_arg_val<uint32_t>(arg_idx++);
    const auto intermediate_base_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sender_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t compute_sync_sem_addr = get_arg_val<uint32_t>(arg_idx++);

    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, num_connections, arg_idx);

    cb_reserve_back(packet_header_cb_id, 1);
    const uint32_t sem_header_addr = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    const uint64_t sender_sem_noc_addr = get_noc_addr(remote_sender_noc_x, remote_sender_noc_y, sender_semaphore_addr);

    auto* sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)sem_header_ptr, sender_num_hops);
    sem_header_ptr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_sem_noc_addr, 1});

    auto& connection = fabric_connection.get(0).sender;
    connection.wait_for_empty_write_slot();
    connection.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr, packet_header_size_bytes);

    // Wait for remote sender to signal data has been written to intermediate tensor
    auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
    noc_semaphore_wait(local_semaphore_ptr, 1);
    noc_semaphore_set(local_semaphore_ptr, 0);  // Reset for next iteration

    close_connections(fabric_connection);

    // Signal all non-data compute cores via NOC multicast semaphore set
    // Each compute_reader will wait locally on its semaphore
    if constexpr (mcast_num_dests > 0) {
        // Set the source semaphore value to 1
        auto compute_sync_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(compute_sync_sem_addr);
        noc_semaphore_set(compute_sync_sem_ptr, 1);

        // Multicast the semaphore value to all non-data compute cores
        uint64_t mcast_sem_addr =
            get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, compute_sync_sem_addr);
        noc_semaphore_set_multicast(compute_sync_sem_addr, mcast_sem_addr, mcast_num_dests);
    }
    // Now both local and remote data are ready, push to compute
    cb_push_back(cb_compute, input_num_tiles);

    cb_push_back(receiver_cb, input_num_tiles);
}
