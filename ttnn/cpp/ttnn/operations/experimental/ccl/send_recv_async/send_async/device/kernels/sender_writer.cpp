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
constexpr uint32_t data_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_pages = get_compile_time_arg_val(2);
constexpr uint32_t socket_block_size = get_compile_time_arg_val(3);  // This is assumed to be aligned
constexpr uint32_t socket_page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_pages_per_packet = get_compile_time_arg_val(5);
// Used when there are multiple pages per packet
constexpr uint32_t num_whole_packets = get_compile_time_arg_val(6);
constexpr uint32_t num_pages_remainder = get_compile_time_arg_val(7);
// Used when there are multiple packets per page
constexpr uint32_t num_whole_packets_per_page = get_compile_time_arg_val(8);
constexpr uint32_t aligned_partial_packet_size = get_compile_time_arg_val(9);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(10);
constexpr bool is_dram = get_compile_time_arg_val(11);

template <uint32_t packet_size, uint32_t cb_id, bool is_dram>
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
    uint32_t socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t bank_id = get_arg_val<uint32_t>(rt_args_idx++);

    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // This kernel relies on two fabric headers stored in fabric_packet_header_cb:
    //  - data_packet_header: Used for issuing writes to downstream data cores
    //  - socket_packet_header: Used by socket APIs for control flow
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));
    fabric_connection.open();

    // Create Socket Interface
    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, socket_block_size);

    fabric_set_unicast_route(data_packet_header_addr, sender_socket);

    uint64_t receiver_noc_coord_addr = get_noc_addr_from_bank_id<is_dram>(bank_id, 0, edm_fabric_write_noc_index);

    if constexpr (num_pages_per_packet > 0) {
        constexpr uint32_t full_packet_size = num_pages_per_packet * socket_page_size;
        constexpr uint32_t remainder_packet_size = num_pages_remainder * socket_page_size;

        for (uint32_t i = 0; i < num_whole_packets; ++i) {
            socket_reserve_pages(sender_socket, 1);
            uint64_t dst_addr = receiver_noc_coord_addr + sender_socket.write_ptr;
            write_data_to_remote_core<full_packet_size, data_cb_id, is_dram>(
                fabric_connection, dst_addr, data_packet_header_addr);
            socket_push_pages(sender_socket, 1);
            fabric_socket_notify_receiver(sender_socket, fabric_connection, socket_packet_header_addr);
        }

        if constexpr (num_pages_remainder > 0) {
            socket_reserve_pages(sender_socket, 1);
            uint64_t dst_addr = receiver_noc_coord_addr + sender_socket.write_ptr;
            write_data_to_remote_core<remainder_packet_size, data_cb_id, is_dram>(
                fabric_connection, dst_addr, data_packet_header_addr);
            socket_push_pages(sender_socket, 1);
            fabric_socket_notify_receiver(sender_socket, fabric_connection, socket_packet_header_addr);
        }

    }
    // Large pages. We pack page chunks into a single packet.
    else {
        for (uint32_t i = 0; i < num_pages; ++i) {
            socket_reserve_pages(sender_socket, 1);
            uint64_t dst_addr = receiver_noc_coord_addr + sender_socket.write_ptr;
            for (uint32_t j = 0; j < num_whole_packets_per_page; ++j) {
                write_data_to_remote_core<whole_packet_size, data_cb_id, is_dram>(
                    fabric_connection, dst_addr, data_packet_header_addr);
                dst_addr += whole_packet_size;
            }
            if constexpr (aligned_partial_packet_size > 0) {
                write_data_to_remote_core<aligned_partial_packet_size, data_cb_id, is_dram>(
                    fabric_connection, dst_addr, data_packet_header_addr);
            }
            socket_push_pages(sender_socket, 1);
            fabric_socket_notify_receiver(sender_socket, fabric_connection, socket_packet_header_addr);
        }
    }
    update_socket_config(sender_socket);
    fabric_connection.close();
}
