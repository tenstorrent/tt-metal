// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/common.hpp"

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

inline void read_from_local(
    uint32_t src_addr_l,
    uint32_t src_addr_s,
    uint32_t src_addr_m,
    uint32_t page_bytes,
    uint32_t core_noc_x,
    uint32_t core_noc_y,
    uint32_t cb_id_in_l,  // compute cb for l
    uint32_t cb_id_in_s,  // compute cb for s
    uint32_t cb_id_in_m,  // compute cb for m
    uint32_t onetile,
    uint32_t input_num_tiles) {
    // read l, s, m data from own device and push it to compute cbs
    cb_reserve_back(cb_id_in_l, input_num_tiles);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in_l);
    uint64_t read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_l);
    noc_async_read(read_addr, l1_write_addr, input_num_tiles * page_bytes);
    cb_push_back(cb_id_in_l, input_num_tiles);

    // for tensor s
    cb_reserve_back(cb_id_in_s, onetile);
    l1_write_addr = get_write_ptr(cb_id_in_s);
    read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_s);
    noc_async_read(read_addr, l1_write_addr, onetile * page_bytes);
    cb_push_back(cb_id_in_s, onetile);

    // for tensor m
    cb_reserve_back(cb_id_in_m, onetile);
    l1_write_addr = get_write_ptr(cb_id_in_m);
    read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_m);
    noc_async_read(read_addr, l1_write_addr, onetile * page_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_id_in_m, onetile);
}

void kernel_main() {
    constexpr uint32_t fabric_ct_idx = get_compile_time_arg_val(0);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t receiver_cb_id_l = get_compile_time_arg_val(3);
    constexpr uint32_t receiver_cb_id_s = get_compile_time_arg_val(4);
    constexpr uint32_t receiver_cb_id_m = get_compile_time_arg_val(5);
    constexpr uint32_t alignment = get_compile_time_arg_val(6);
    constexpr uint32_t compute_cb_l = get_compile_time_arg_val(7);
    constexpr uint32_t compute_cb_s = get_compile_time_arg_val(8);
    constexpr uint32_t compute_cb_m = get_compile_time_arg_val(9);
    constexpr uint32_t input_num_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t packet_size_bytes = get_compile_time_arg_val(12);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(fabric_ct_idx);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(fabric_ct_idx + 1);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(fabric_ct_idx + 2);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(fabric_ct_idx + 3);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(fabric_ct_idx + 4);

    size_t arg_idx = 0;
    const uint32_t fabric_idx_2 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t src_addr_l = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t src_addr_s = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t src_addr_m = get_arg_val<uint32_t>(arg_idx++);
    const auto intermediate_base_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sender_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sender_semaphore_addr2 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_noc_y = get_arg_val<uint32_t>(arg_idx++);

    const uint8_t sender_num_hops = 1;

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
    const uint32_t sem_header_addr = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    const uint64_t sender_sem_noc_addr = get_noc_addr(core_noc_x, core_noc_y, sender_semaphore_addr);

    auto* sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)sem_header_ptr, sender_num_hops);

    sem_header_ptr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_sem_noc_addr, 1});

    mux_connection.wait_for_empty_write_slot();
    mux_connection.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr, packet_header_size_bytes);

    cb_reserve_back(packet_cb_id, 1);
    uint32_t packet_l1_addr = get_write_ptr(packet_cb_id);

    // read local data from own device and push to compute cbs
    read_from_local(
        src_addr_l,
        src_addr_s,
        src_addr_m,
        page_size_bytes,
        core_noc_x,
        core_noc_y,
        compute_cb_l,
        compute_cb_s,
        compute_cb_m,
        1,
        input_num_tiles);

    //  receive l, s and m data from sender
    auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
    noc_semaphore_wait(local_semaphore_ptr, 1);
    noc_semaphore_set(local_semaphore_ptr, 0);

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

    const uint32_t aligned_page_size_bytes = align(page_size_bytes, alignment);
    uint32_t packet_idx = 0;

    cb_reserve_back(receiver_cb_id_l, input_num_tiles);
    uint32_t dest_page_base_addr = get_write_ptr(receiver_cb_id_l);

    uint64_t packet_noc_addr = get_noc_addr(core_noc_x, core_noc_y, intermediate_base_addr);
    noc_async_read(packet_noc_addr, packet_l1_addr, new_packet_size_bytes);

    // moving l tensor
    tt_memmove<true, false, false, 0>(dest_page_base_addr, packet_l1_addr, packet_size_bytes);
    cb_push_back(receiver_cb_id_l, input_num_tiles);
    //  now s and m
    cb_reserve_back(receiver_cb_id_s, 1);
    cb_reserve_back(receiver_cb_id_m, 1);

    uint32_t dest_page_base_addr_s = get_write_ptr(receiver_cb_id_s);
    tt_memmove<true, false, false, 0>(
        dest_page_base_addr_s, packet_l1_addr + packet_size_bytes, aligned_page_size_bytes);
    cb_push_back(receiver_cb_id_s, 1);

    uint32_t dest_page_base_addr_m = get_write_ptr(receiver_cb_id_m);
    tt_memmove<true, false, false, 0>(
        dest_page_base_addr_m, packet_l1_addr + packet_size_bytes + aligned_page_size_bytes, aligned_page_size_bytes);
    cb_push_back(receiver_cb_id_m, 1);

    cb_push_back(packet_cb_id, 1);

    noc_async_read_barrier();

    // now the similar behaviour when device 2 is sending data to device 1
    // will be waiting on another semaphore, and fabric is for the other 2 muxes
    size_t fabric_idx_2_ref = fabric_idx_2;
    const bool is_termination_master2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const uint8_t fabric_mux_x2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const uint8_t fabric_mux_y2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const size_t fabric_mux_channel_base_address2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const size_t fabric_mux_connection_info_address2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const size_t fabric_mux_connection_handshake_address2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const size_t fabric_mux_flow_control_address2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const size_t fabric_mux_buffer_index_address2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const uint8_t fabric_mux_channel_id2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);

    uint32_t termination_sync_address2 = get_semaphore(get_arg_val<uint32_t>(fabric_idx_2_ref++));
    uint32_t local_fabric_mux_status_address2 = get_semaphore(get_arg_val<uint32_t>(fabric_idx_2_ref++));
    uint32_t local_flow_control_address2 = get_semaphore(get_arg_val<uint32_t>(fabric_idx_2_ref++));
    uint32_t local_teardown_address2 = get_semaphore(get_arg_val<uint32_t>(fabric_idx_2_ref++));
    uint32_t local_buffer_index_address2 = get_semaphore(get_arg_val<uint32_t>(fabric_idx_2_ref++));

    uint32_t termination_master_noc_x2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    uint32_t termination_master_noc_y2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);

    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* mux_connection_handle2;
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> mux_connection2;

    mux_connection2 = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
        fabric_mux_x2,
        fabric_mux_y2,
        fabric_mux_channel_id2,
        fabric_mux_num_buffers_per_channel,
        fabric_mux_channel_buffer_size_bytes,
        fabric_mux_channel_base_address2,
        fabric_mux_connection_info_address2,
        fabric_mux_connection_handshake_address2,
        fabric_mux_flow_control_address2,
        fabric_mux_buffer_index_address2,
        local_flow_control_address2,
        local_teardown_address2,
        local_buffer_index_address2);
    mux_connection_handle2 = &mux_connection2;

    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x2, fabric_mux_y2, fabric_mux_status_address, local_fabric_mux_status_address);
    tt::tt_fabric::fabric_client_connect(*mux_connection_handle2);

    cb_reserve_back(packet_header_cb_id, 1);
    const uint32_t sem_header_addr_2 = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    const uint64_t sender_sem_noc_addr_2 = get_noc_addr(core_noc_x, core_noc_y, sender_semaphore_addr2);
    auto* sem_header_ptr_2 = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr_2);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)sem_header_ptr_2, sender_num_hops);
    sem_header_ptr_2->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_sem_noc_addr_2, 1});

    mux_connection2.wait_for_empty_write_slot();
    mux_connection2.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr_2, packet_header_size_bytes);

    // read again l, s and m from device 2
    local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr2);
    noc_semaphore_wait(local_semaphore_ptr, 1);
    noc_semaphore_set(local_semaphore_ptr, 0);
    tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle2);

    if (is_termination_master) {
        auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address2);
        noc_semaphore_wait(termination_sync_ptr, num_mux_clients - 1);
        tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x2, fabric_mux_y2, fabric_mux_termination_signal_address);
    } else {
        uint64_t dest_addr =
            safe_get_noc_addr(termination_master_noc_x2, termination_master_noc_y2, termination_sync_address2, 0);
        noc_semaphore_inc(dest_addr, 1);
        noc_async_atomic_barrier();
    }

    cb_reserve_back(packet_cb_id, 1);
    packet_l1_addr = get_write_ptr(packet_cb_id);

    packet_idx = 0;

    cb_reserve_back(receiver_cb_id_l, input_num_tiles);
    dest_page_base_addr = get_write_ptr(receiver_cb_id_l);
    packet_noc_addr = get_noc_addr(core_noc_x, core_noc_y, intermediate_base_addr);
    noc_async_read(packet_noc_addr, packet_l1_addr, new_packet_size_bytes);

    tt_memmove<true, false, false, 0>(dest_page_base_addr, packet_l1_addr, packet_size_bytes);
    cb_push_back(receiver_cb_id_l, input_num_tiles);

    cb_reserve_back(receiver_cb_id_s, 1);
    cb_reserve_back(receiver_cb_id_m, 1);
    dest_page_base_addr_s = get_write_ptr(receiver_cb_id_s);
    dest_page_base_addr_m = get_write_ptr(receiver_cb_id_m);

    tt_memmove<true, false, false, 0>(
        dest_page_base_addr_s, packet_l1_addr + packet_size_bytes, aligned_page_size_bytes);
    tt_memmove<true, false, false, 0>(
        dest_page_base_addr_m, packet_l1_addr + packet_size_bytes + aligned_page_size_bytes, aligned_page_size_bytes);
    cb_push_back(receiver_cb_id_s, 1);
    cb_push_back(receiver_cb_id_m, 1);
    cb_push_back(packet_cb_id, 1);

    noc_async_read_barrier();
}
