// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include <cstdint>
#include <utility>
#include "tt_metal/fabric/hw/inc/linear/api.h"

using tt::data_movement::common::round_up;

void kernel_main() {
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t alignment = get_compile_time_arg_val(2);
    constexpr uint32_t input_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t payload_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t core_noc_x = get_compile_time_arg_val(6);
    constexpr uint32_t core_noc_y = get_compile_time_arg_val(7);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    const uint32_t receiver_base_address = get_arg_val<uint32_t>(0);
    const uint32_t receive_semaphore_addr = get_arg_val<uint32_t>(1);
    const bool dst_is_forward = get_arg_val<uint32_t>(2);

    const uint8_t dst_num_hops = 1;

    size_t conn_arg_idx = 2;

    auto fabric_connection = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(conn_arg_idx);

    cb_reserve_back(packet_header_cb_id, 1);
    uint32_t packet_header_addr = get_read_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    auto* packet_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)packet_header_ptr, dst_num_hops);

    //  wait for receiver to signal it is ready
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive_semaphore_addr), 1);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive_semaphore_addr), 0);

    fabric_connection.open_finish();
    auto& connection_direction =
        tt::point_to_point::common::connection_direction_collection(dst_is_forward, fabric_connection);

    cb_wait_front(packet_cb_id, 1);
    uint32_t packet_base_addr = get_write_ptr(packet_cb_id);

    const uint64_t dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, receiver_base_address);
    const uint64_t receive_sem_noc_addr = get_noc_addr(core_noc_x, core_noc_y, receive_semaphore_addr);

    // Use fused packet API to send data + semaphore increment in a single packet
    packet_header_ptr->to_noc_fused_unicast_write_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, receive_sem_noc_addr, 1, true},
        align(payload_size_bytes, alignment));

    connection_direction.wait_for_empty_write_slot();
    connection_direction.send_payload_without_header_non_blocking_from_address(packet_base_addr, payload_size_bytes);
    connection_direction.send_payload_flush_blocking_from_address(
        (uint32_t)packet_header_ptr, sizeof(PACKET_HEADER_TYPE));

    cb_pop_front(packet_cb_id, 1);

    fabric_connection.close();
}
