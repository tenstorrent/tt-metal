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
using tt::data_movement::common::tt_memmove;

void kernel_main() {
    constexpr uint32_t num_tiles_l = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t packet_cb_id0 = get_compile_time_arg_val(2);

    size_t arg_idx = 0;
    const uint32_t src_addr_l = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t src_addr_s = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t src_addr_m = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t current_core_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t current_core_y = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t onetile = 1;

    // ROUND 1: read local data
    cb_reserve_back(packet_cb_id0, 1);
    uint32_t l1_write_addr = get_write_ptr(packet_cb_id0);
    uint64_t read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_l);
    noc_async_read(read_addr, l1_write_addr, num_tiles_l * page_bytes);
    read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_s);
    noc_async_read(read_addr, l1_write_addr + num_tiles_l * page_bytes, onetile * page_bytes);
    read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_m);
    noc_async_read(read_addr, l1_write_addr + (num_tiles_l + onetile) * page_bytes, onetile * page_bytes);
    noc_async_read_barrier();
    cb_push_back(packet_cb_id0, 1);

    constexpr uint32_t fabric_ct_idx = get_compile_time_arg_val(3);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t receiver_cb_id_l = get_compile_time_arg_val(6);
    constexpr uint32_t receiver_cb_id_s = get_compile_time_arg_val(7);
    constexpr uint32_t receiver_cb_id_m = get_compile_time_arg_val(8);
    constexpr uint32_t alignment = get_compile_time_arg_val(9);
    constexpr uint32_t compute_cb_l = get_compile_time_arg_val(10);
    constexpr uint32_t compute_cb_s = get_compile_time_arg_val(11);
    constexpr uint32_t compute_cb_m = get_compile_time_arg_val(12);
    constexpr uint32_t input_num_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t packet_size_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t round1_interm_cb_id = get_compile_time_arg_val(16);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(fabric_ct_idx);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(fabric_ct_idx + 1);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(fabric_ct_idx + 2);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(fabric_ct_idx + 3);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(fabric_ct_idx + 4);

    const auto pkt_base_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sender_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    // ROUND 2: receive data from neighbour device and send to compute
    uint32_t round1_interm_tensor_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_semaphore = get_arg_val<uint32_t>(arg_idx++);

    const uint8_t sender_num_hops = 1;
    const uint32_t aligned_page_size_bytes = align(page_size_bytes, alignment);

    const uint32_t new_packet_size_bytes = packet_size_bytes + 2 * align(page_size_bytes, alignment);

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

    // Barrier optimization: only one core sends the fabric barrier
    const bool is_barrier_leader = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t writer2_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t writer2_noc_y = get_arg_val<uint32_t>(arg_idx++);

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

    cb_reserve_back(packet_header_cb_id, 1);
    const uint32_t sem_header_addr_2 = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    // Only barrier leader sends the fabric semaphore to remote writer
    // The remote writer will mcast to other local writers
    if (is_barrier_leader) {
        const uint64_t sender_sem_noc_addr_2 = get_noc_addr(current_core_x, current_core_y, sender_semaphore_addr);
        auto* sem_header_ptr_2 = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr_2);
        fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)sem_header_ptr_2, sender_num_hops);
        sem_header_ptr_2->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_sem_noc_addr_2, 1});

        mux_connection.wait_for_empty_write_slot();
        mux_connection.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr_2, packet_header_size_bytes);
    }

    // wait for device semaphore
    auto device_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(device_semaphore);
    noc_semaphore_wait(device_semaphore_ptr, 1);
    noc_semaphore_set(device_semaphore_ptr, 0);

    uint32_t round1_l1_write_addr = get_write_ptr(round1_interm_cb_id);

    noc_async_read(
        get_noc_addr(writer2_noc_x, writer2_noc_y, round1_interm_tensor_addr),
        round1_l1_write_addr,
        new_packet_size_bytes);

    noc_async_read_barrier();

    cb_reserve_back(compute_cb_l, input_num_tiles);
    cb_reserve_back(compute_cb_s, 1);
    cb_reserve_back(compute_cb_m, 1);

    uint32_t compute_cb_l_addr = get_write_ptr(compute_cb_l);
    uint32_t compute_cb_s_addr = get_write_ptr(compute_cb_s);
    uint32_t compute_cb_m_addr = get_write_ptr(compute_cb_m);

    tt_memmove<true, false, false, 0>(compute_cb_l_addr, round1_l1_write_addr, packet_size_bytes);
    tt_memmove<true, false, false, 0>(
        compute_cb_s_addr, round1_l1_write_addr + packet_size_bytes, aligned_page_size_bytes);
    tt_memmove<true, false, false, 0>(
        compute_cb_m_addr, round1_l1_write_addr + packet_size_bytes + aligned_page_size_bytes, aligned_page_size_bytes);

    cb_push_back(compute_cb_l, input_num_tiles);
    cb_push_back(compute_cb_s, 1);
    cb_push_back(compute_cb_m, 1);

    // read again l, s and m from device 2
    volatile tt_l1_ptr uint32_t* local_semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
    noc_semaphore_wait(local_semaphore_ptr, 1);
    noc_semaphore_set(local_semaphore_ptr, 0);

    cb_reserve_back(packet_cb_id, 1);
    uint32_t packet_l1_addr = get_write_ptr(packet_cb_id);

    cb_reserve_back(receiver_cb_id_l, input_num_tiles);
    cb_reserve_back(receiver_cb_id_s, 1);
    cb_reserve_back(receiver_cb_id_m, 1);

    uint32_t dest_page_base_addr = get_write_ptr(receiver_cb_id_l);
    uint32_t dest_page_base_addr_s = get_write_ptr(receiver_cb_id_s);
    uint32_t dest_page_base_addr_m = get_write_ptr(receiver_cb_id_m);

    uint64_t packet_noc_addr = get_noc_addr(current_core_x, current_core_y, pkt_base_addr);

    noc_async_read(packet_noc_addr, packet_l1_addr, new_packet_size_bytes);
    noc_async_read_barrier();  // Wait for Round 2 data to be read before memmove

    tt_memmove<true, false, false, 0>(dest_page_base_addr, packet_l1_addr, packet_size_bytes);
    tt_memmove<true, false, false, 0>(
        dest_page_base_addr_s, packet_l1_addr + packet_size_bytes, aligned_page_size_bytes);
    tt_memmove<true, false, false, 0>(
        dest_page_base_addr_m, packet_l1_addr + packet_size_bytes + aligned_page_size_bytes, aligned_page_size_bytes);

    cb_push_back(receiver_cb_id_l, input_num_tiles);
    cb_push_back(receiver_cb_id_s, 1);
    cb_push_back(receiver_cb_id_m, 1);

    // Write Round 2 received data to data core's intermediate shard
    uint64_t bw_interm_data_core_addr = get_noc_addr(core_noc_x, core_noc_y, pkt_base_addr);
    noc_async_write(packet_l1_addr, bw_interm_data_core_addr, new_packet_size_bytes);
    noc_async_write_barrier();
    cb_push_back(packet_cb_id, 1);

    // Disconnect from mux
    tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle);

    // Reader1 terminates its mux (backward for D0/D2, forward for D1/D3)
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
