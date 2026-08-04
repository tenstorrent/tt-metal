// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/operations/experimental/ccl/send_recv_async/buffered_common/buffered_async_types.hpp"
///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t data_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t handshake_page_size = get_compile_time_arg_val(2);
constexpr uint32_t output_page_size = get_compile_time_arg_val(3);
constexpr uint32_t socket_page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_pages_per_packet = get_compile_time_arg_val(5);
constexpr uint32_t num_whole_packets_per_page = get_compile_time_arg_val(6);
constexpr uint32_t partial_packet_size = get_compile_time_arg_val(7);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(8);
constexpr uint32_t num_banks = get_compile_time_arg_val(9);
constexpr uint32_t enable_bank_packing = get_compile_time_arg_val(10);
constexpr uint32_t output_args_cta_idx = 11;
constexpr uint32_t output_args_crta_idx = 0;

FORCE_INLINE void fabric_write_page(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr,
    uint32_t l1_read_addr,
    uint64_t dst_noc_addr,
    uint32_t size_bytes) {
    data_packet_header_addr->to_noc_unicast_write(NocUnicastCommandHeader{dst_noc_addr}, size_bytes);
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_without_header_non_blocking_from_address(l1_read_addr, size_bytes);
    fabric_connection.send_payload_flush_blocking_from_address(
        (uint32_t)data_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t rt_args_idx = 0;
    uint32_t socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_pages = get_arg_val<uint32_t>(rt_args_idx++);          // pages for this core
    uint32_t page_start_offset = get_arg_val<uint32_t>(rt_args_idx++);  // page start offset for this core
    [[maybe_unused]] uint32_t num_whole_packets = get_arg_val<uint32_t>(rt_args_idx++);    // whole packets (fallback)
    [[maybe_unused]] uint32_t num_pages_remainder = get_arg_val<uint32_t>(rt_args_idx++);  // remainder (fallback)
    uint32_t handshake_base_addr = get_arg_val<uint32_t>(rt_args_idx++);

    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // Separate headers so the data path and the socket control path do not clobber each other.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            get_write_ptr(fabric_packet_header_cb_id) + sizeof(PACKET_HEADER_TYPE));
    fabric_connection.open();

    // Create Socket Interface
    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, handshake_page_size);

    // Only one downstream in this op
    sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, 0);
    fabric_set_unicast_route(data_packet_header_addr, downstream_enc);

    // Ring state lives in the persistent L1_SMALL buffer at handshake_base_addr, so it survives
    // across runs. A non-zero num_tensors means the receiver already advertised the ring and the
    // handshake can be skipped; the host zeroes this buffer when it builds the program.
    volatile tt_l1_ptr OutputTensorInfo* dest_info =
        reinterpret_cast<volatile tt_l1_ptr OutputTensorInfo*>(handshake_base_addr);

    //////////////////////////////////////////////////
    // STEP 1: advertise the handshake-buffer address over the socket
    //////////////////////////////////////////////////
    if (dest_info->num_tensors == 0) {
        socket_reserve_pages(sender_socket, 1);
        uint64_t advertise_dst_addr = get_noc_addr(
            downstream_enc.d2d.downstream_noc_x,
            downstream_enc.d2d.downstream_noc_y,
            sender_socket.downstream_fifo_addr + sender_socket.write_ptr);

        data_packet_header_addr->to_noc_unicast_inline_write(
            NocUnicastInlineWriteCommandHeader{advertise_dst_addr, handshake_base_addr});
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_flush_blocking_from_address(
            (uint32_t)data_packet_header_addr, sizeof(PACKET_HEADER_TYPE));

        socket_push_pages(sender_socket, 1);
        fabric_socket_notify_receiver(sender_socket, fabric_connection, socket_packet_header_addr);

        //////////////////////////////////////////////////
        // STEP 2: wait for the receiver to write back the ring of receive-buffer base addresses
        //////////////////////////////////////////////////
        do {
            invalidate_l1_cache();
        } while (dest_info->num_tensors == 0);
        update_socket_config(sender_socket);
    }

    // Block while the ring is full. The indices are monotonic counters, so the modulo below is what
    // maps them onto a buffer.
    do {
        invalidate_l1_cache();
    } while ((dest_info->write_index[0] - dest_info->read_index[0]) >= dest_info->num_tensors);

    uint32_t write_index = dest_info->write_index[0];
    uint32_t output_buffer_index = write_index % dest_info->num_tensors;
    uint32_t output_base_addr = dest_info->base_addr[output_buffer_index];

    dest_info->write_index[0] = write_index + 1;

    auto output_addr_gen_args = TensorAccessorArgs<output_args_cta_idx, output_args_crta_idx>();
    auto output_addr_gen = TensorAccessor(output_addr_gen_args, output_base_addr, output_page_size);
    //////////////////////////////////////////////////
    // STEP 3: stream pages directly into the selected receiver output tensor
    //////////////////////////////////////////////////
    if constexpr (enable_bank_packing) {
        // Bank-contiguous packing, see send_direct_async/kernels/sender_direct_writer.cpp.
        constexpr uint32_t super_block_pages = num_banks * num_pages_per_packet;
        constexpr uint32_t bank_region_bytes = num_pages_per_packet * output_page_size;
        const uint32_t end_page = page_start_offset + num_pages;
        for (uint32_t sb_base = page_start_offset; sb_base < end_page; sb_base += super_block_pages) {
            cb_wait_front(data_cb_id, 1);
            const uint32_t l1_base = get_read_ptr(data_cb_id);
            for (uint32_t b = 0; b < num_banks; ++b) {
                const uint32_t head = sb_base + b;
                if (head >= end_page) {
                    break;  // remaining banks in this super-block have no pages
                }
                uint32_t count = 0;
                for (uint32_t pp = head; count < num_pages_per_packet && pp < end_page; pp += num_banks) {
                    ++count;
                }
                fabric_write_page(
                    fabric_connection,
                    data_packet_header_addr,
                    l1_base + b * bank_region_bytes,
                    output_addr_gen.get_noc_addr(head),
                    count * output_page_size);
            }
            cb_pop_front(data_cb_id, 1);
        }
    } else if constexpr (num_pages_per_packet > 0) {
        // Small pages: each CB entry holds num_pages_per_packet whole pages at socket_page_size stride.
        uint32_t page_index = page_start_offset;
        for (uint32_t i = 0; i < num_whole_packets; ++i) {
            cb_wait_front(data_cb_id, 1);
            uint32_t l1_read_addr = get_read_ptr(data_cb_id);
            for (uint32_t j = 0; j < num_pages_per_packet; ++j) {
                uint64_t out_noc_addr = output_addr_gen.get_noc_addr(page_index);
                fabric_write_page(
                    fabric_connection, data_packet_header_addr, l1_read_addr, out_noc_addr, output_page_size);
                l1_read_addr += socket_page_size;
                page_index++;
            }
            cb_pop_front(data_cb_id, 1);
        }

        if (num_pages_remainder > 0) {
            cb_wait_front(data_cb_id, 1);
            uint32_t l1_read_addr = get_read_ptr(data_cb_id);
            for (uint32_t j = 0; j < num_pages_remainder; ++j) {
                uint64_t out_noc_addr = output_addr_gen.get_noc_addr(page_index);
                fabric_write_page(
                    fabric_connection, data_packet_header_addr, l1_read_addr, out_noc_addr, output_page_size);
                l1_read_addr += socket_page_size;
                page_index++;
            }
            cb_pop_front(data_cb_id, 1);
        }
    } else {
        // Large pages: each output page spans multiple fabric packets (one CB entry per packet).
        uint32_t page_index = page_start_offset;
        for (uint32_t i = 0; i < num_pages; ++i) {
            uint64_t out_noc_addr = output_addr_gen.get_noc_addr(page_index);
            for (uint32_t j = 0; j < num_whole_packets_per_page; ++j) {
                cb_wait_front(data_cb_id, 1);
                uint32_t l1_read_addr = get_read_ptr(data_cb_id);
                fabric_write_page(
                    fabric_connection, data_packet_header_addr, l1_read_addr, out_noc_addr, whole_packet_size);
                cb_pop_front(data_cb_id, 1);
                out_noc_addr += whole_packet_size;
            }
            if constexpr (partial_packet_size > 0) {
                cb_wait_front(data_cb_id, 1);
                uint32_t l1_read_addr = get_read_ptr(data_cb_id);
                fabric_write_page(
                    fabric_connection, data_packet_header_addr, l1_read_addr, out_noc_addr, partial_packet_size);
                cb_pop_front(data_cb_id, 1);
            }
            page_index++;
        }
    }

    //////////////////////////////////////////////////
    // STEP 4: tell the receiver a buffer is filled by bumping its copy of write_index
    //////////////////////////////////////////////////
    uint32_t write_l1_addr = dest_info->receiver_config_l1_addr + offsetof(OutputTensorInfo, write_index);
    uint64_t write_noc_addr =
        get_noc_addr(downstream_enc.d2d.downstream_noc_x, downstream_enc.d2d.downstream_noc_y, write_l1_addr);
    fabric_set_unicast_route(socket_packet_header_addr, downstream_enc);
    socket_packet_header_addr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{write_noc_addr, 1});
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_flush_blocking_from_address(
        (uint32_t)socket_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
    fabric_connection.close();
}
