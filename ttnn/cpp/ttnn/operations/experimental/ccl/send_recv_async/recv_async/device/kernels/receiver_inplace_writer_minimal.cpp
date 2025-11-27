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
constexpr uint32_t output_page_size = get_compile_time_arg_val(1);  // This is assumed to be aligned
constexpr uint32_t socket_block_size = get_compile_time_arg_val(2);
constexpr uint32_t num_pages_per_ack = get_compile_time_arg_val(3);
constexpr uint32_t output_args_cta_idx = 4;
constexpr uint32_t output_args_crta_idx = 0;

FORCE_INLINE void notify_sender(
    SocketReceiverInterface& receiver_socket,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr) {
    noc_async_writes_flushed();
    fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);
}

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

    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    fabric_connection.open();

    // Create Socket Interface
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, socket_block_size);

    auto output_addr_gen_args = TensorAccessorArgs<output_args_cta_idx, output_args_crta_idx>();
    auto output_addr_gen = TensorAccessor(output_addr_gen_args, output_base_addr, output_page_size);
    bool ack_sent = false;
    for (uint32_t i = 0; i < 100; ++i) {
        auto noc_write_addr = output_addr_gen.get_noc_addr(0);
        socket_wait_for_pages(receiver_socket, 1);
        uint32_t l1_read_addr = receiver_socket.read_ptr;
        uint32_t val = 0;
        for (uint32_t j = 0; j < output_page_size / 4; j += 4) {
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_read_addr + j) != val) {
                while (true);
            }
            val++;
        }
        noc_async_write<output_page_size>(receiver_socket.read_ptr, noc_write_addr, output_page_size);
        socket_pop_pages(receiver_socket, 1);
        ack_sent = false;
        if (i % num_pages_per_ack == 0) {
            notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);
            ack_sent = true;
        }
    }

    if (!ack_sent) {
        notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);
        ack_sent = true;
    }

    update_socket_config(receiver_socket);
    fabric_connection.close();
}
