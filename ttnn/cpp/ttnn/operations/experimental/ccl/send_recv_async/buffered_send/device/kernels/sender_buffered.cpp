// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/debug/dprint.h"  // required in all kernels using DPRINT
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
constexpr uint32_t output_args_cta_idx = 9;
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
    uint32_t num_pages = get_arg_val<uint32_t>(rt_args_idx++);            // pages for this core
    uint32_t page_start_offset = get_arg_val<uint32_t>(rt_args_idx++);    // page start offset for this core
    uint32_t num_whole_packets = get_arg_val<uint32_t>(rt_args_idx++);    // whole packets for this core
    uint32_t num_pages_remainder = get_arg_val<uint32_t>(rt_args_idx++);  // remainder pages for this core
    // Base address of the persistent L1_SMALL handshake buffer: page 0 is the dest-info landing
    // zone (the receiver writes the OutputTensorInfo struct here), page 1 stages the advertise
    // payload.
    uint32_t handshake_base_addr = get_arg_val<uint32_t>(rt_args_idx++);

    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // Two fabric headers stored in fabric_packet_header_cb:
    //  - data_packet_header: issues direct writes to the receiver (output tensor + handshake advertise)
    //  - socket_packet_header: used by socket APIs for control flow
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

    // Handshake buffer pages: page 0 is the dest-info landing zone, page 1 stages the advertise
    // payload. Both live in the persistent L1_SMALL buffer addressed by handshake_base_addr.

    // The receiver signals validity by writing the OutputTensorInfo struct, whose first field
    // (num_tensors) becomes non-zero once the struct lands. Clear it before advertising so we never
    // observe a stale completion.
    auto* dest_info_ptr = reinterpret_cast<volatile tt_l1_ptr OutputTensorInfo*>(handshake_base_addr);

    DPRINT("Output tensor info size = {}\n", sizeof(OutputTensorInfo));
    //////////////////////////////////////////////////
    // STEP 1: advertise the handshake-buffer address over the socket
    //////////////////////////////////////////////////
    DPRINT("handshake_base_addr = {}, handshake_page_size = {}\n", handshake_base_addr, handshake_page_size);
    if (dest_info_ptr->num_tensors == 0) {
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

        // fabric_write_page(
        //     fabric_connection, data_packet_header_addr, advertise_stage_addr, advertise_dst_addr,
        //     handshake_page_size);
        socket_push_pages(sender_socket, 1);
        fabric_socket_notify_receiver(sender_socket, fabric_connection, socket_packet_header_addr);
        DPRINT("Sent handshake address to receiver\n");

        //////////////////////////////////////////////////
        // STEP 2: wait for the receiver to write back the destination tensor info, including the ring of
        // receive-buffer base addresses.
        //
        // SKELETON: the data is streamed into the first receive buffer only. The N-buffer ring and the
        // global-semaphore-based coordination still need to be implemented.
        //////////////////////////////////////////////////
        DPRINT("Waiting for destination tensor info from receiver in {}\n", handshake_base_addr);
        do {
            invalidate_l1_cache();
        } while (dest_info_ptr->num_tensors == 0);
        DPRINT("Received destination tensor info from receiver\n");
        update_socket_config(sender_socket);
    } else {
        DPRINT("Destination tensor info already received from receiver\n");
    }

    // Copy the whole OutputTensorInfo struct out of the landing zone exactly once, then work from
    // the local copy (the ring of receive-buffer base addresses lives in dest_info.base_addr).
    volatile OutputTensorInfo* dest_info = reinterpret_cast<volatile tt_l1_ptr OutputTensorInfo*>(handshake_base_addr);
    DPRINT("num_output_buffers = {}\n", dest_info->num_tensors);
    uint32_t output_base_addr = dest_info->base_addr[dest_info->write_index[0]];

    DPRINT("Writing to output tensor index {}\n", dest_info->write_index[0]);

    dest_info->write_index[0] = (dest_info->write_index[0] + 1) % dest_info->num_tensors;

    DPRINT("Updated write index to {}\n", dest_info->write_index[0]);

    auto output_addr_gen_args = TensorAccessorArgs<output_args_cta_idx, output_args_crta_idx>();
    auto output_addr_gen = TensorAccessor(output_addr_gen_args, output_base_addr, output_page_size);
    //////////////////////////////////////////////////
    // STEP 3: stream pages directly into the receiver's (first) output tensor
    //////////////////////////////////////////////////
    uint32_t page_index = page_start_offset;
    if constexpr (num_pages_per_packet > 0) {
        // Small pages: each CB entry holds num_pages_per_packet whole pages at socket_page_size stride.
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
    DPRINT("Streamed pages directly into the receiver's (first) output tensor\n");
    dest_info->read_index[0] = dest_info->read_index[0] % dest_info->num_tensors;
    DPRINT("On sender: Write Index = {}, Read Index = {}\n", dest_info->write_index[0], dest_info->read_index[0]);

    //////////////////////////////////////////////////
    // STEP 4: push a single completion page onto the socket
    //////////////////////////////////////////////////
    uint32_t write_l1_addr = dest_info->receiver_config_l1_addr + offsetof(OutputTensorInfo, write_index);
    DPRINT("Incrementing addr {}\n", write_l1_addr);
    uint64_t write_noc_addr =
        get_noc_addr(downstream_enc.d2d.downstream_noc_x, downstream_enc.d2d.downstream_noc_y, write_l1_addr);
    fabric_set_unicast_route(socket_packet_header_addr, downstream_enc);
    socket_packet_header_addr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{write_noc_addr, 1});
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_flush_blocking_from_address(
        (uint32_t)socket_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
    fabric_connection.close();
    DPRINT("Closed fabric connection\n");
}
