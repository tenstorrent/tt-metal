// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "dataflow_api.h"
#include "socket_api.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t socket_block_size = get_compile_time_arg_val(1);
constexpr uint32_t num_whole_packets_per_page = get_compile_time_arg_val(2);
constexpr uint32_t aligned_partial_packet_size = get_compile_time_arg_val(3);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(4);

template <uint32_t packet_size, uint32_t cb_id>
FORCE_INLINE void write_data_to_remote_core(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    uint64_t dst_addr,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr) {
    cb_wait_front(cb_id, 1);
    auto l1_read_addr = get_read_ptr(cb_id);
    data_packet_header_addr->to_noc_unicast_write(NocUnicastCommandHeader{dst_addr}, packet_size);
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_without_header_non_blocking_from_address(l1_read_addr, packet_size);
    fabric_connection.send_payload_flush_blocking_from_address(
        (uint32_t)data_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
    cb_pop_front(cb_id, 1);
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    // Setup Fabric Headers and Connections
    size_t rt_args_idx = 0;
    uint32_t send_socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t recv_socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t downstream_bank_id = get_arg_val<uint32_t>(rt_args_idx++);

    tt::tt_fabric::WorkerToFabricEdmSender upstream_fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    tt::tt_fabric::WorkerToFabricEdmSender downstream_fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // This kernel relies on three fabric headers stored in fabric_packet_header_cb:
    //  - downstream_data_packet_header: Used for issuing writes to downstream data cores
    //  - socket_packet_header: Used by socket APIs for control flow
    volatile tt_l1_ptr PACKET_HEADER_TYPE* upstream_socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* downstream_data_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + 2 * sizeof(PACKET_HEADER_TYPE));

    upstream_fabric_connection.open();
    downstream_fabric_connection.open();

    SocketSenderInterface send_socket = create_sender_socket_interface(send_socket_config_addr);
    SocketReceiverInterface recv_socket = create_receiver_socket_interface(recv_socket_config_addr);

    set_sender_socket_page_size(send_socket, socket_block_size);
    set_receiver_socket_page_size(recv_socket, socket_block_size);

    // Only one downstream in this op
    sender_downstream_encoding downstream_enc = get_downstream_encoding(send_socket, 0);
    fabric_set_unicast_route(downstream_data_packet_header_addr, downstream_enc);

    uint64_t receiver_noc_coord_addr = get_noc_addr_from_bank_id<false>(
        downstream_bank_id, 0, tt::tt_fabric::connection_interface::edm_fabric_write_noc_index);

    // Reserve pages downstream and wait for pages to come from upstream
    socket_reserve_pages(send_socket, 1);
    socket_wait_for_pages(recv_socket, 1);

    auto l1_read_addr = recv_socket.read_ptr;
    uint64_t dst_addr = receiver_noc_coord_addr + send_socket.write_ptr;

    // Forward data to downstream
    for (uint32_t j = 0; j < num_whole_packets_per_page; ++j) {
        downstream_data_packet_header_addr->to_noc_unicast_write(NocUnicastCommandHeader{dst_addr}, whole_packet_size);
        downstream_fabric_connection.wait_for_empty_write_slot();
        downstream_fabric_connection.send_payload_without_header_non_blocking_from_address(
            l1_read_addr, whole_packet_size);
        downstream_fabric_connection.send_payload_flush_blocking_from_address(
            (uint32_t)downstream_data_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
        dst_addr += whole_packet_size;
        l1_read_addr += whole_packet_size;
    }
    if constexpr (aligned_partial_packet_size > 0) {
        downstream_data_packet_header_addr->to_noc_unicast_write(
            NocUnicastCommandHeader{dst_addr}, aligned_partial_packet_size);
        downstream_fabric_connection.wait_for_empty_write_slot();
        downstream_fabric_connection.send_payload_without_header_non_blocking_from_address(
            l1_read_addr, aligned_partial_packet_size);
        downstream_fabric_connection.send_payload_flush_blocking_from_address(
            (uint32_t)downstream_data_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
    }

    // Notify Upstream and Downstream that data has been consumed or produced
    socket_pop_pages(recv_socket, 1);
    socket_push_pages(send_socket, 1);
    fabric_socket_notify_receiver(send_socket, downstream_fabric_connection, downstream_socket_packet_header_addr);
    fabric_socket_notify_sender(recv_socket, upstream_fabric_connection, upstream_socket_packet_header_addr);

    update_socket_config(send_socket);
    update_socket_config(recv_socket);
    upstream_fabric_connection.close();
    downstream_fabric_connection.close();
}
