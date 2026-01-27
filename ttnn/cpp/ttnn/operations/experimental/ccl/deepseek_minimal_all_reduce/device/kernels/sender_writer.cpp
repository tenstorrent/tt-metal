// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include <cstdint>
#include <utility>
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/api_common.h"

using namespace tt::tt_fabric::linear::experimental;
using namespace tt::tt_fabric::common::experimental;

using tt::data_movement::common::round_up;

void kernel_main() {
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t alignment = get_compile_time_arg_val(2);
    constexpr uint32_t input_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t payload_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t data_noc_x = get_compile_time_arg_val(6);
    constexpr uint32_t data_noc_y = get_compile_time_arg_val(7);
    constexpr uint32_t remote_receiver_noc_x = get_compile_time_arg_val(8);
    constexpr uint32_t remote_receiver_noc_y = get_compile_time_arg_val(9);
    constexpr uint32_t dst_num_hops = get_compile_time_arg_val(10);
    constexpr uint32_t num_connections = get_compile_time_arg_val(11);
    constexpr bool using_persistent_buffer = get_compile_time_arg_val(12);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    size_t arg_idx = 0;
    const uint32_t receiver_base_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t receive_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);

    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, num_connections, arg_idx);

    cb_reserve_back(packet_header_cb_id, 1);
    uint32_t packet_header_addr = get_read_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    auto* packet_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
    fabric_set_unicast_route(fabric_connection, packet_header_ptr, 0);
    packet_header_ptr->to_chip_unicast(dst_num_hops);

    //  wait for receiver to signal it is ready
    if constexpr (!using_persistent_buffer) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive_semaphore_addr), 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive_semaphore_addr), 0);
    }

    cb_wait_front(packet_cb_id, input_num_tiles);
    uint32_t packet_base_addr = get_read_ptr(packet_cb_id);

    const uint64_t dst_noc_addr = get_noc_addr(data_noc_x, data_noc_y, receiver_base_address);
    const uint64_t receive_sem_noc_addr =
        get_noc_addr(remote_receiver_noc_x, remote_receiver_noc_y, receive_semaphore_addr);

    // Use fused packet API to send data + semaphore increment in a single packet
    packet_header_ptr->to_noc_fused_unicast_write_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, receive_sem_noc_addr, 1, true},
        align(payload_size_bytes, alignment));

    auto& connection = fabric_connection.get(0).sender;
    connection.wait_for_empty_write_slot();
    connection.send_payload_without_header_non_blocking_from_address(packet_base_addr, payload_size_bytes);
    connection.send_payload_flush_blocking_from_address((uint32_t)packet_header_ptr, sizeof(PACKET_HEADER_TYPE));

    cb_pop_front(packet_cb_id, input_num_tiles);

    close_connections(fabric_connection);
}
