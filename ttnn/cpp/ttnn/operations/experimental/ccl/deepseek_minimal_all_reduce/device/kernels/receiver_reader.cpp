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
    DPRINT << "start of receiver reader kernel\n";
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

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
    constexpr uint8_t sender_num_hops = 1;

    size_t arg_idx = 0;
    uint32_t tensor_address0 = get_arg_val<uint32_t>(arg_idx++);
    const auto intermediate_base_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sender_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_connections = get_arg_val<uint32_t>(arg_idx++);

    DPRINT << "compile time args:\n";
    DPRINT << " packet_header_cb_id: " << (uint32_t)packet_header_cb_id << "\n";
    DPRINT << " receiver_cb: " << (uint32_t)receiver_cb << "\n";
    DPRINT << " alignment: " << (uint32_t)alignment << "\n";
    DPRINT << " cb_compute: " << (uint32_t)cb_compute << "\n";
    DPRINT << " input_num_tiles: " << (uint32_t)input_num_tiles << "\n";
    DPRINT << " page_size_bytes: " << (uint32_t)page_size_bytes << "\n";
    DPRINT << " packet_size_bytes: " << (uint32_t)packet_size_bytes << "\n";
    DPRINT << " data_noc_x: " << (uint32_t)data_noc_x << "\n";
    DPRINT << " data_noc_y: " << (uint32_t)data_noc_y << "\n";
    DPRINT << " remote_sender_noc_x: " << (uint32_t)remote_sender_noc_x << "\n";
    DPRINT << " remote_sender_noc_y: " << (uint32_t)remote_sender_noc_y << "\n";

    DPRINT << "arg vals:\n";
    DPRINT << " tensor_address0: " << (uint32_t)tensor_address0 << "\n";
    DPRINT << " intermediate_base_addr: " << (uint32_t)intermediate_base_addr << "\n";
    DPRINT << " sender_semaphore_addr: " << (uint32_t)sender_semaphore_addr << "\n";
    DPRINT << " num_connections: " << (uint32_t)num_connections << "\n";

    DPRINT << "before building fabric connection\n";
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, num_connections, arg_idx);

    DPRINT << "creating fabric connection\n";
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

    DPRINT << "before reading from local tensor\n";
    //  read local data from own device and push to compute cbs
    cb_reserve_back(cb_compute, input_num_tiles);
    const uint32_t l1_write_addr = get_write_ptr(cb_compute);
    uint64_t base_src_addr = get_noc_addr(data_noc_x, data_noc_y, tensor_address0);
    noc_async_read(base_src_addr, l1_write_addr, input_num_tiles * page_size_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_compute, input_num_tiles);
    // end of read local data
    DPRINT << "after reading from local tensor\n";

    auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
    noc_semaphore_wait(local_semaphore_ptr, 1);
    noc_semaphore_set(local_semaphore_ptr, 0);

    DPRINT << "after semaphore wait\n";

    close_connections(fabric_connection);

    cb_reserve_back(receiver_cb, input_num_tiles);
    const uint32_t packet_l1_addr = get_write_ptr(receiver_cb);
    const uint64_t packet_noc_addr = get_noc_addr(data_noc_x, data_noc_y, intermediate_base_addr);
    noc_async_read(packet_noc_addr, packet_l1_addr, packet_size_bytes);
    noc_async_read_barrier();
    cb_push_back(receiver_cb, input_num_tiles);
    DPRINT << "end of receiver reader kernel\n";
}
