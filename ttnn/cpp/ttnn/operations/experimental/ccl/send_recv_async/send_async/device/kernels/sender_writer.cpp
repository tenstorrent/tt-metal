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
    uint32_t socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t bank_id = get_arg_val<uint32_t>(rt_args_idx++);

    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection_2 =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
    // This kernel relies on two fabric headers stored in fabric_packet_header_cb:
    //  - data_packet_header: Used for issuing writes to downstream data cores
    //  - socket_packet_header: Used by socket APIs for control flow
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr_2 =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + 2 * sizeof(PACKET_HEADER_TYPE));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr_2 =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + 3 * sizeof(PACKET_HEADER_TYPE));

    fabric_connection.open();
    fabric_connection_2.open();
    // Create Socket Interface
    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, socket_block_size);
    DPRINT << "Sender Page Size: " << socket_block_size << ENDL();
    // Only one downstream in this op
    sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);
    fabric_set_unicast_route(data_packet_header_addr, downstream_enc);
    fabric_set_unicast_route(data_packet_header_addr_2, downstream_enc);

    uint64_t receiver_noc_coord_addr =
        get_noc_addr_from_bank_id<is_dram>(bank_id, 0, tt::tt_fabric::connection_interface::edm_fabric_write_noc_index);

    if constexpr (num_pages_per_packet > 0) {
        constexpr uint32_t full_packet_size = num_pages_per_packet * socket_page_size;
        constexpr uint32_t remainder_packet_size = num_pages_remainder * socket_page_size;

        for (uint32_t i = 0; i < num_whole_packets; ++i) {
            socket_reserve_pages(sender_socket, 1);
            uint64_t dst_addr = receiver_noc_coord_addr + sender_socket.write_ptr;
            write_data_to_remote_core<full_packet_size, data_cb_id>(
                fabric_connection, dst_addr, data_packet_header_addr);
            socket_push_pages(sender_socket, 1);
            fabric_socket_notify_receiver(sender_socket, fabric_connection, socket_packet_header_addr);
        }

        if constexpr (num_pages_remainder > 0) {
            socket_reserve_pages(sender_socket, 1);
            uint64_t dst_addr = receiver_noc_coord_addr + sender_socket.write_ptr;
            write_data_to_remote_core<remainder_packet_size, data_cb_id>(
                fabric_connection, dst_addr, data_packet_header_addr);
            socket_push_pages(sender_socket, 1);
            fabric_socket_notify_receiver(sender_socket, fabric_connection, socket_packet_header_addr);
        }

    }
    // Large pages. We pack page chunks into a single packet.
    else {
        uint64_t downstream_bytes_sent_noc_addr = get_noc_addr(
            downstream_enc.downstream_noc_x, downstream_enc.downstream_noc_y, sender_socket.downstream_bytes_sent_addr);
        for (int i = 0; i < 1000000; i++) {
            DPRINT << "Reserving page:" << i << ENDL();
            socket_reserve_pages(sender_socket, 1);
            DPRINT << "Done reserving page:" << i << ENDL();
            cb_wait_front(data_cb_id, 1);

            auto l1_read_addr = get_read_ptr(data_cb_id);
            uint64_t dst_addr = receiver_noc_coord_addr + sender_socket.write_ptr;
            // Send first 2 packets on Routing Plane 0
            for (uint32_t j = 0; j < 2; ++j) {
                data_packet_header_addr->to_noc_fused_unicast_write_atomic_inc(
                    NocUnicastAtomicIncFusedCommandHeader{dst_addr, downstream_bytes_sent_noc_addr, whole_packet_size},
                    whole_packet_size);
                fabric_connection.wait_for_empty_write_slot();
                fabric_connection.send_payload_without_header_non_blocking_from_address(
                    l1_read_addr, whole_packet_size);
                fabric_connection.send_payload_flush_blocking_from_address(
                    (uint32_t)data_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
                dst_addr += whole_packet_size;
                l1_read_addr += whole_packet_size;
            }
            data_packet_header_addr_2->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{dst_addr, downstream_bytes_sent_noc_addr, whole_packet_size},
                whole_packet_size);
            fabric_connection_2.wait_for_empty_write_slot();
            fabric_connection_2.send_payload_without_header_non_blocking_from_address(l1_read_addr, whole_packet_size);
            fabric_connection_2.send_payload_flush_blocking_from_address(
                (uint32_t)data_packet_header_addr_2, sizeof(PACKET_HEADER_TYPE));
            dst_addr += whole_packet_size;
            l1_read_addr += whole_packet_size;

            data_packet_header_addr_2->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{
                    dst_addr, downstream_bytes_sent_noc_addr, aligned_partial_packet_size},
                aligned_partial_packet_size);
            fabric_connection_2.wait_for_empty_write_slot();
            fabric_connection_2.send_payload_without_header_non_blocking_from_address(
                l1_read_addr, aligned_partial_packet_size);
            fabric_connection_2.send_payload_flush_blocking_from_address(
                (uint32_t)data_packet_header_addr_2, sizeof(PACKET_HEADER_TYPE));
            cb_pop_front(data_cb_id, 1);
            socket_push_pages(sender_socket, 1);
        }
    }

    update_socket_config(sender_socket);
    fabric_connection.close();
    fabric_connection_2.close();
}
