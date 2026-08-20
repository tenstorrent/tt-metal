// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t handshake_page_size = get_compile_time_arg_val(1);

// direct_dest_info layout (must match send_direct_async/device/kernels/sender_direct_writer.cpp).
constexpr uint32_t DEST_VALID_OFFSET = 0;
constexpr uint32_t DEST_OUTPUT_ADDR_OFFSET = 4;
constexpr uint32_t DEST_PAGE_SIZE_OFFSET = 8;
constexpr uint32_t DEST_NUM_PAGES_OFFSET = 12;

FORCE_INLINE void fabric_inline_write_upstream(
    const SocketReceiverInterface& receiver_socket,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header_addr,
    uint64_t dst_noc_addr,
    uint32_t value) {
    fabric_set_unicast_route(packet_header_addr, receiver_socket);
    packet_header_addr->to_noc_unicast_inline_write(NocUnicastInlineWriteCommandHeader{dst_noc_addr, value});
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_flush_blocking_from_address(
        (uint32_t)packet_header_addr, sizeof(PACKET_HEADER_TYPE));
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t rt_args_idx = 0;
    uint32_t socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_base_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_page_size = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_pages = get_arg_val<uint32_t>(rt_args_idx++);

    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    fabric_connection.open();

    // Create Socket Interface
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, handshake_page_size);

    //////////////////////////////////////////////////
    // STEP 1: receive the sender's advertised handshake-buffer address
    //////////////////////////////////////////////////
    socket_wait_for_pages(receiver_socket, 1);
    uint32_t sender_handshake_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_socket.read_ptr)[0];
    socket_pop_pages(receiver_socket, 1);
    fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);

    //////////////////////////////////////////////////
    // STEP 2: write the destination tensor info back into the sender's handshake buffer.
    // Write the data fields first, then the valid flag last so the sender never observes a
    // partially-written struct.
    //////////////////////////////////////////////////
    uint32_t upstream_noc_x = receiver_socket.d2d.upstream_noc_x;
    uint32_t upstream_noc_y = receiver_socket.d2d.upstream_noc_y;

    fabric_inline_write_upstream(
        receiver_socket,
        fabric_connection,
        socket_packet_header_addr,
        get_noc_addr(upstream_noc_x, upstream_noc_y, sender_handshake_addr + DEST_OUTPUT_ADDR_OFFSET),
        output_base_addr);
    fabric_inline_write_upstream(
        receiver_socket,
        fabric_connection,
        socket_packet_header_addr,
        get_noc_addr(upstream_noc_x, upstream_noc_y, sender_handshake_addr + DEST_PAGE_SIZE_OFFSET),
        output_page_size);
    fabric_inline_write_upstream(
        receiver_socket,
        fabric_connection,
        socket_packet_header_addr,
        get_noc_addr(upstream_noc_x, upstream_noc_y, sender_handshake_addr + DEST_NUM_PAGES_OFFSET),
        num_pages);
    fabric_inline_write_upstream(
        receiver_socket,
        fabric_connection,
        socket_packet_header_addr,
        get_noc_addr(upstream_noc_x, upstream_noc_y, sender_handshake_addr + DEST_VALID_OFFSET),
        1);

    //////////////////////////////////////////////////
    // STEP 3: wait for the completion token from the sender
    //////////////////////////////////////////////////
    socket_wait_for_pages(receiver_socket, 1);
    socket_pop_pages(receiver_socket, 1);
    fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);

    update_socket_config(receiver_socket);
    fabric_connection.close();
}
