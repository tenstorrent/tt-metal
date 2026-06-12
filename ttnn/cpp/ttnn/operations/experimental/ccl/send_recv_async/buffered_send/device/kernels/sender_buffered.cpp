// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/debug/dprint.h"  // required in all kernels using DPRINT
///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t data_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t handshake_cb_id = get_compile_time_arg_val(2);
constexpr uint32_t handshake_page_size = get_compile_time_arg_val(3);
constexpr uint32_t output_page_size = get_compile_time_arg_val(4);
constexpr uint32_t socket_page_size = get_compile_time_arg_val(5);
constexpr uint32_t num_pages_per_packet = get_compile_time_arg_val(6);
constexpr uint32_t num_whole_packets_per_page = get_compile_time_arg_val(7);
constexpr uint32_t partial_packet_size = get_compile_time_arg_val(8);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(9);
constexpr uint32_t output_args_cta_idx = 10;
constexpr uint32_t output_args_crta_idx = 0;

// Upper bound on the size of the receiver's ring of receive buffers. Only used to size a scratch
// array; the actual count is advertised by the receiver at runtime (INFO_NUM_TENSORS_OFFSET).
constexpr uint32_t MAX_OUTPUT_TENSORS = 12;

// Handshake landing-zone layout (must match buffered_recv/device/kernels/receiver_buffered.cpp).
// Offsets are in bytes from the base of handshake page 0. The receiver bulk-writes the
// OutputTensorInfo struct at INFO_STRUCT_OFFSET (kept 0 so the NoC write destination is aligned),
// then sets the 4-byte valid flag at the end of the page last so the sender never observes a
// partially-written struct.
constexpr uint32_t INFO_STRUCT_OFFSET = 0;
// OutputTensorInfo struct fields (relative to handshake page 0):
constexpr uint32_t INFO_NUM_TENSORS_OFFSET = INFO_STRUCT_OFFSET + 0;
constexpr uint32_t INFO_PAGE_SIZE_OFFSET = INFO_STRUCT_OFFSET + 4;
constexpr uint32_t INFO_NUM_PAGES_OFFSET = INFO_STRUCT_OFFSET + 8;
// Array of `num_tensors` output base addresses (one per receive buffer in the ring).
constexpr uint32_t INFO_BASE_ADDR_OFFSET = INFO_STRUCT_OFFSET + 12;
// Valid sentinel lives in the last word of the handshake page so it never overlaps the struct and
// is written by a separate (4-byte aligned) inline write after the struct lands.
constexpr uint32_t INFO_VALID_OFFSET = handshake_page_size - sizeof(uint32_t);

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

    // Handshake buffer: page 0 is the dest-info landing zone, page 1 stages the advertise payload.
    uint32_t handshake_base_addr = get_write_ptr(handshake_cb_id);
    uint32_t advertise_stage_addr = handshake_base_addr + handshake_page_size;

    // Clear the valid flag before advertising so we never observe a stale completion.
    volatile tt_l1_ptr uint32_t* valid_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_base_addr);
    *valid_ptr = 0;

    // Stage the address the receiver should write its dest-info struct back to.
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(advertise_stage_addr)[0] = handshake_base_addr;

    //////////////////////////////////////////////////
    // STEP 1: advertise the handshake-buffer address over the socket
    //////////////////////////////////////////////////
    DPRINT("advertise_stage_addr = {}\n", advertise_stage_addr);
    {
        socket_reserve_pages(sender_socket, 1);
        uint64_t advertise_dst_addr = get_noc_addr(
            downstream_enc.d2d.downstream_noc_x,
            downstream_enc.d2d.downstream_noc_y,
            sender_socket.downstream_fifo_addr + sender_socket.write_ptr);
        fabric_write_page(
            fabric_connection, data_packet_header_addr, advertise_stage_addr, advertise_dst_addr, handshake_page_size);
        socket_push_pages(sender_socket, 1);
        fabric_socket_notify_receiver(sender_socket, fabric_connection, socket_packet_header_addr);
    }
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
    } while (*valid_ptr == 0);
    DPRINT("Received destination tensor info from receiver\n");
    uint32_t num_output_buffers =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_base_addr + INFO_NUM_TENSORS_OFFSET)[0];
    DPRINT("num_output_buffers = {}\n", num_output_buffers);
    // Receive all advertised output base addresses (the ring of receive buffers).
    uint32_t output_base_addrs[MAX_OUTPUT_TENSORS];
    for (uint32_t i = 0; i < num_output_buffers; ++i) {
        output_base_addrs[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            handshake_base_addr + INFO_BASE_ADDR_OFFSET + i * sizeof(uint32_t))[0];
    }
    DPRINT("output_base_addrs = {}\n", output_base_addrs);
    uint32_t output_base_addr = output_base_addrs[0];
    // DPRINT("output_base_addr = {}\n", output_base_addr);
    auto output_addr_gen_args = TensorAccessorArgs<output_args_cta_idx, output_args_crta_idx>();
    auto output_addr_gen = TensorAccessor(output_addr_gen_args, output_base_addr, output_page_size);
    //////////////////////////////////////////////////
    // STEP 3: stream pages directly into the receiver's (first) output tensor
    //////////////////////////////////////////////////
    DPRINT("Streaming pages directly into the receiver's (first) output tensor\n");
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
    //////////////////////////////////////////////////
    // STEP 4: push a single completion page onto the socket
    //////////////////////////////////////////////////
    socket_reserve_pages(sender_socket, 1);
    socket_push_pages(sender_socket, 1);
    fabric_socket_notify_receiver(sender_socket, fabric_connection, socket_packet_header_addr);
    DPRINT("Sent completion page to receiver\n");
    update_socket_config(sender_socket);
    fabric_connection.close();
    DPRINT("Closed fabric connection\n");
}
