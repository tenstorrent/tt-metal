// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "dataflow_api.h"
#include "socket_api.h"
#include "debug/dprint.h"
///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t data_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t socket_block_size = get_compile_time_arg_val(2);  // This is assumed to be aligned
constexpr uint32_t aligned_partial_packet_size = get_compile_time_arg_val(3);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(4);
constexpr uint32_t num_whole_packets_link_0 = get_compile_time_arg_val(5);
constexpr uint32_t num_whole_packets_link_1 = get_compile_time_arg_val(6);

void write_data_to_remote_core_with_ack(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header_addr,
    uint32_t l1_read_addr,
    uint64_t dst_addr,
    uint64_t downstream_bytes_sent_noc_addr,
    uint32_t packet_size) {
    packet_header_addr->to_noc_fused_unicast_write_atomic_inc(
        NocUnicastAtomicIncFusedCommandHeader{dst_addr, downstream_bytes_sent_noc_addr, packet_size}, packet_size);
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_without_header_non_blocking_from_address(l1_read_addr, packet_size);
    fabric_connection.send_payload_flush_blocking_from_address(
        (uint32_t)packet_header_addr, sizeof(PACKET_HEADER_TYPE));
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    // Setup Fabric Headers and Connections
    size_t rt_args_idx = 0;
    uint32_t socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t bank_id = get_arg_val<uint32_t>(rt_args_idx++);

    // Hardcoded to use two routing planes
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection_2 =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // One packet header per routing plane
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr_2 =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + 2 * sizeof(PACKET_HEADER_TYPE));

    fabric_connection.open();
    fabric_connection_2.open();

    // Create Socket Interface
    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, socket_block_size);

    // Only one downstream in this op
    sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);
    fabric_set_unicast_route(data_packet_header_addr, downstream_enc);
    fabric_set_unicast_route(data_packet_header_addr_2, downstream_enc);

    uint64_t receiver_noc_coord_addr =
        get_noc_addr_from_bank_id<false>(bank_id, 0, tt::tt_fabric::connection_interface::edm_fabric_write_noc_index);

    uint64_t downstream_bytes_sent_noc_addr = get_noc_addr(
        downstream_enc.downstream_noc_x, downstream_enc.downstream_noc_y, sender_socket.downstream_bytes_sent_addr);
    for (int i = 0; i < 100; i++) {
        DPRINT << "Reserving page:" << i << ENDL();
        socket_reserve_pages(sender_socket, 1);
        DPRINT << "Done reserving page:" << i << ENDL();
        cb_wait_front(data_cb_id, 1);

        auto l1_read_addr = get_read_ptr(data_cb_id);
        uint64_t dst_addr = receiver_noc_coord_addr + sender_socket.write_ptr;

        // Split sending packets between routing planes
        for (uint32_t j = 0; j < num_whole_packets_link_0; ++j) {
            write_data_to_remote_core_with_ack(
                fabric_connection,
                data_packet_header_addr,
                l1_read_addr,
                dst_addr,
                downstream_bytes_sent_noc_addr,
                whole_packet_size);

            dst_addr += whole_packet_size;
            l1_read_addr += whole_packet_size;
        }

        for (uint32_t j = 0; j < num_whole_packets_link_1; ++j) {
            write_data_to_remote_core_with_ack(
                fabric_connection_2,
                data_packet_header_addr_2,
                l1_read_addr,
                dst_addr,
                downstream_bytes_sent_noc_addr,
                whole_packet_size);

            dst_addr += whole_packet_size;
            l1_read_addr += whole_packet_size;
        }

        write_data_to_remote_core_with_ack(
            fabric_connection_2,
            data_packet_header_addr_2,
            l1_read_addr,
            dst_addr,
            downstream_bytes_sent_noc_addr,
            aligned_partial_packet_size);

        cb_pop_front(data_cb_id, 1);
        socket_push_pages(sender_socket, 1);
    }
    update_socket_config(sender_socket);
    fabric_connection.close();
    fabric_connection_2.close();
}
