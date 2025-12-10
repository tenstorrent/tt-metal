// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "dataflow_api.h"
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
using tt::data_movement::common::tt_memmove;

void kernel_main() {
    constexpr uint32_t fabric_ct_idx = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_l = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_s = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_m = get_compile_time_arg_val(3);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t alignment = get_compile_time_arg_val(6);
    constexpr uint32_t input_num_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t payload_size_bytes = get_compile_time_arg_val(9);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(fabric_ct_idx);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(fabric_ct_idx + 1);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(fabric_ct_idx + 2);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(fabric_ct_idx + 3);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(fabric_ct_idx + 4);

    size_t arg_idx = 0;
    const uint32_t receiver_base_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t receive_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t dst_num_hops = 1;
    const uint32_t aligned_page_size_bytes = round_up(page_size_bytes, alignment);

    const bool is_termination_master = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_channel_id = get_arg_val<uint32_t>(arg_idx++);

    uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);

    // device 2 writer receives data from compute kernel and sends it to device 1

    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* mux_connection_handle;
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> mux_connection;

    mux_connection = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
        fabric_mux_x,
        fabric_mux_y,
        fabric_mux_channel_id,
        fabric_mux_num_buffers_per_channel,
        fabric_mux_channel_buffer_size_bytes,
        fabric_mux_channel_base_address,
        fabric_mux_connection_info_address,
        fabric_mux_connection_handshake_address,
        fabric_mux_flow_control_address,
        fabric_mux_buffer_index_address,
        local_flow_control_address,
        local_teardown_address,
        local_buffer_index_address);
    mux_connection_handle = &mux_connection;
    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);

    tt::tt_fabric::fabric_client_connect(*mux_connection_handle);

    // set up packet header buffer
    cb_reserve_back(packet_header_cb_id, 1);
    uint32_t packet_header_addr = get_read_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    auto* packet_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)packet_header_ptr, dst_num_hops);

    const uint32_t new_payload_size_bytes =
        payload_size_bytes + 2 * aligned_page_size_bytes;  // add the extra size for s and m

    cb_reserve_back(packet_cb_id, 1);
    const uint32_t packet_base_addr = get_write_ptr(packet_cb_id);

    uint32_t packet_idx = 0;
    //  wait for receiver to signal it is ready
    auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive_semaphore_addr);
    noc_semaphore_wait_min(local_semaphore_ptr, 1);
    noc_semaphore_set(local_semaphore_ptr, 0);
    cb_wait_front(cb_id_l, input_num_tiles);
    uint32_t src_page_base_addr = get_read_ptr(cb_id_l);
    tt_memmove<true, false, false, 0>(packet_base_addr, src_page_base_addr, payload_size_bytes);
    cb_pop_front(cb_id_l, input_num_tiles);

    cb_wait_front(cb_id_s, 1);
    const uint32_t src_page_base_addr_s = get_read_ptr(cb_id_s);
    tt_memmove<true, false, false, 0>(
        packet_base_addr + payload_size_bytes, src_page_base_addr_s, aligned_page_size_bytes);
    cb_pop_front(cb_id_s, 1);

    cb_wait_front(cb_id_m, 1);
    const uint32_t src_page_base_addr_m = get_read_ptr(cb_id_m);
    tt_memmove<true, false, false, 0>(
        packet_base_addr + payload_size_bytes + aligned_page_size_bytes, src_page_base_addr_m, aligned_page_size_bytes);
    cb_pop_front(cb_id_m, 1);
    cb_push_back(packet_cb_id, 1);

    const uint64_t dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, receiver_base_address);
    const uint64_t receive_sem_noc_addr = get_noc_addr(core_noc_x, core_noc_y, receive_semaphore_addr);

    // Use fused packet API to send data + semaphore increment in a single packet
    packet_header_ptr->to_noc_fused_unicast_write_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, receive_sem_noc_addr, 1, true},
        align(new_payload_size_bytes, alignment));

    mux_connection.wait_for_empty_write_slot();
    mux_connection.send_payload_without_header_non_blocking_from_address(packet_base_addr, new_payload_size_bytes);
    mux_connection.send_payload_flush_blocking_from_address((uint32_t)packet_header_ptr, sizeof(PACKET_HEADER_TYPE));

    tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle);
    if (is_termination_master) {
        auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address);
        noc_semaphore_wait(termination_sync_ptr, num_mux_clients - 1);
        tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
    } else {
        uint64_t dest_addr =
            safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
        noc_semaphore_inc(dest_addr, 1);
        noc_async_atomic_barrier();
    }
}
