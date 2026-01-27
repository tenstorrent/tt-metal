// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);  // CB for remote data (intermediate tensor)
    constexpr uint32_t alignment = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in2 = get_compile_time_arg_val(3);  // CB for local data (input tensor)
    constexpr uint32_t remote_sender_noc_x = get_compile_time_arg_val(4);
    constexpr uint32_t remote_sender_noc_y = get_compile_time_arg_val(5);
    constexpr uint32_t num_standard_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t cb_residual = get_compile_time_arg_val(7);  // CB for residual tensor (optional)
    constexpr uint32_t has_residual = get_compile_time_arg_val(8);
    constexpr bool using_persistent_buffer = get_compile_time_arg_val(9);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
    constexpr uint8_t sender_num_hops = 1;
    constexpr uint32_t num_connections = 1;

    size_t arg_idx = 0;
    const uint32_t sender_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);

    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, num_connections, arg_idx);

    cb_reserve_back(packet_header_cb_id, 1);
    const uint32_t sem_header_addr = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    const uint64_t sender_sem_noc_addr = get_noc_addr(remote_sender_noc_x, remote_sender_noc_y, sender_semaphore_addr);

    if constexpr (!using_persistent_buffer) {
        auto* sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr);
        fabric_set_unicast_route(fabric_connection, sem_header_ptr, 0);
        sem_header_ptr->to_chip_unicast(sender_num_hops);

        sem_header_ptr->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_sem_noc_addr, 1});

        auto& connection = fabric_connection.get(0).sender;
        connection.wait_for_empty_write_slot();
        connection.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr, packet_header_size_bytes);
    }

    // Push local and residual tiles to compute immediately (they're ready)
    // This allows compute to start (local + residual) while waiting for remote data
    cb_reserve_back(cb_in2, num_standard_tiles);
    cb_push_back(cb_in2, num_standard_tiles);

    if constexpr (has_residual) {
        cb_reserve_back(cb_residual, num_standard_tiles);
        cb_push_back(cb_residual, num_standard_tiles);
    }

    // Wait for remote sender to signal data has been written to intermediate tensor
    auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
    noc_semaphore_wait(local_semaphore_ptr, 1);
    noc_semaphore_set(local_semaphore_ptr, 0);

    close_connections(fabric_connection);

    // Remote data is now ready, push to compute
    cb_reserve_back(cb_in1, num_standard_tiles);
    cb_push_back(cb_in1, num_standard_tiles);
}
