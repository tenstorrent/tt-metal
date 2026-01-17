// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/common.hpp"

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include <cstdint>
#include <utility>
#include "tt_metal/fabric/hw/inc/linear/api.h"

using tt::data_movement::common::tt_memmove;

void kernel_main() {
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t receiver_cb = get_compile_time_arg_val(1);
    constexpr uint32_t alignment = get_compile_time_arg_val(2);
    constexpr uint32_t cb_compute = get_compile_time_arg_val(3);
    constexpr uint32_t input_num_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t packet_size_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t core_noc_x = get_compile_time_arg_val(7);
    constexpr uint32_t core_noc_y = get_compile_time_arg_val(8);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    uint32_t tensor_address0 = get_arg_val<uint32_t>(0);
    const auto intermediate_base_addr = get_arg_val<uint32_t>(1);
    const uint32_t sender_semaphore_addr = get_arg_val<uint32_t>(2);
    const bool sender_is_forward = get_arg_val<uint32_t>(3);

    const uint8_t sender_num_hops = 1;

    auto fabric_connection = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(conn_arg_idx);

    cb_reserve_back(packet_header_cb_id, 1);
    const uint32_t sem_header_addr = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    const uint64_t sender_sem_noc_addr = get_noc_addr(core_noc_x, core_noc_y, sender_semaphore_addr);

    auto* sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)sem_header_ptr, sender_num_hops);
    sem_header_ptr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_sem_noc_addr, 1});

    fabric_connection.open_finish();
    auto& connection_direction =
        tt::point_to_point::common::connection_direction_collection(sender_is_forward, fabric_connection);
    connection_direction.wait_for_empty_write_slot();
    connection_direction.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr, packet_header_size_bytes);

    //  read local data from own device and push to compute cbs
    cb_reserve_back(cb_compute, input_num_tiles);
    const uint32_t l1_write_addr = get_write_ptr(cb_compute);
    uint64_t base_src_addr = get_noc_addr(core_noc_x, core_noc_y, tensor_address0);
    noc_async_read(base_src_addr, l1_write_addr, input_num_tiles * page_size_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_compute, input_num_tiles);
    // end of read local data

    auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
    noc_semaphore_wait(local_semaphore_ptr, 1);
    noc_semaphore_set(local_semaphore_ptr, 0);

    fabric_connection.close();

    cb_reserve_back(receiver_cb, 1);
    const uint32_t packet_l1_addr = get_write_ptr(receiver_cb);
    const uint64_t packet_noc_addr = get_noc_addr(core_noc_x, core_noc_y, intermediate_base_addr);
    noc_async_read(packet_noc_addr, packet_l1_addr, packet_size_bytes);
    noc_async_read_barrier();

    tt_memmove<true, false, false, 0>(dest_page_base_addr, packet_l1_addr, packet_size_bytes);
    cb_push_back(receiver_cb, input_num_tiles);
}
