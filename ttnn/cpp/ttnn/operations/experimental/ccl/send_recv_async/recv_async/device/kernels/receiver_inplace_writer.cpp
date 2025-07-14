// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "dataflow_api.h"
#include "socket_api.h"
#include "tt_metal/hw/inc/accessor/tensor_accessor.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t num_pages = get_compile_time_arg_val(1);
constexpr uint32_t output_page_size = get_compile_time_arg_val(2);  // This is assumed to be aligned
constexpr uint32_t socket_block_size = get_compile_time_arg_val(3);
constexpr uint32_t socket_page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_pages_per_packet = get_compile_time_arg_val(5);
// Used when there are multiple pages per packet
constexpr uint32_t num_whole_packets = get_compile_time_arg_val(6);
constexpr uint32_t num_pages_remainder = get_compile_time_arg_val(7);
constexpr uint32_t output_args_cta_idx = 8;
constexpr uint32_t output_args_crta_idx = 0;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    // Setup Fabric Headers and Connections
    size_t rt_args_idx = 0;
    uint32_t socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_base_addr = get_arg_val<uint32_t>(rt_args_idx++);

    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // This kernel relies on two fabric headers stored in fabric_packet_header_cb:
    //  - data_packet_header: Used for issuing reads from upstream data cores
    //  - socket_packet_header: Used by socket APIs for control flow
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    fabric_connection.open();

    // Create Socket Interface
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, socket_block_size);

    auto output_addr_gen_args = make_tensor_accessor_args<output_args_cta_idx, output_args_crta_idx>();
    auto output_addr_gen = make_tensor_accessor_from_args(output_addr_gen_args, output_base_addr, output_page_size);

    // Small pages. We write multiple pages from a single packet.
    uint32_t page_index = 0;
    if constexpr (num_pages_per_packet > 0) {
        for (uint32_t i = 0; i < num_whole_packets; ++i) {
            socket_wait_for_pages(receiver_socket, 1);
            uint32_t l1_read_addr = receiver_socket.read_ptr;
            for (uint32_t j = 0; j < num_pages_per_packet; ++j) {
                auto noc_write_addr = output_addr_gen.get_noc_addr(page_index);
                noc_async_write<output_page_size>(l1_read_addr, noc_write_addr, output_page_size);
                page_index++;
                l1_read_addr += socket_page_size;
            }
            socket_pop_pages(receiver_socket, 1);
            noc_async_writes_flushed();
            fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);
        }

        if (num_pages_remainder > 0) {
            socket_wait_for_pages(receiver_socket, 1);
            uint32_t l1_read_addr = receiver_socket.read_ptr;
            for (uint32_t j = 0; j < num_pages_remainder; ++j) {
                auto noc_write_addr = output_addr_gen.get_noc_addr(page_index);
                noc_async_write<output_page_size>(l1_read_addr, noc_write_addr, output_page_size);
                page_index++;
                l1_read_addr += socket_page_size;
            }
            socket_pop_pages(receiver_socket, 1);
            noc_async_writes_flushed();
            fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);
        }

    }
    // Large pages. We write page chunks from multiple packets.
    else {
        for (uint32_t i = 0; i < num_pages; ++i) {
            auto noc_write_addr = output_addr_gen.get_noc_addr(page_index);
            socket_wait_for_pages(receiver_socket, 1);
            noc_async_write<output_page_size>(receiver_socket.read_ptr, noc_write_addr, output_page_size);
            page_index++;
            socket_pop_pages(receiver_socket, 1);
            noc_async_writes_flushed();
            fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);
        }
    }
    update_socket_config(receiver_socket);
    fabric_connection.close();
}
