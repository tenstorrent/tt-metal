// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);
constexpr BufferType intermediate_type = static_cast<BufferType>(get_compile_time_arg_val(3));
constexpr BufferType output_type = static_cast<BufferType>(get_compile_time_arg_val(4));
constexpr uint32_t cb_forward_id = get_compile_time_arg_val(5);
constexpr uint32_t cb_backward_id = get_compile_time_arg_val(6);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(7);
constexpr uint32_t intermediate_page_size = get_compile_time_arg_val(8);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(9);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(10);
constexpr bool dynamic_alternate = get_compile_time_arg_val(11);
constexpr bool fuse_op = get_compile_time_arg_val(12);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(13));
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(14);
constexpr uint32_t num_inputs = get_compile_time_arg_val(15);

constexpr uint32_t N_DRAM_BANKS = 12;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    uint32_t input_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t gather_dim = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_batch_head_count = get_arg_val<uint32_t>(arg_idx++);
    uint32_t slice_num_pages = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_forward = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_backward = get_arg_val<uint32_t>(arg_idx++);
    address_t intermediate_addresses[num_inputs];
    address_t output_addresses[num_inputs];
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        address_t intermediate_address = get_arg_val<address_t>(arg_idx++);
        address_t output_address = get_arg_val<address_t>(arg_idx++);
        intermediate_addresses[input_idx] = intermediate_address;
        output_addresses[input_idx] = output_address;
    }
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);

    /* Args for overlapped all gather */
    OpSignaler op_signaler_sender;

    if constexpr (fuse_op) {
        arg_idx = arg_for_fab;
        op_signaler_sender = OpSignaler(arg_idx);
    }

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
    pkt_hdr_forward->to_chip_unicast(1);
    pkt_hdr_backward->to_chip_unicast(1);

    // interleaved addrgen
    constexpr bool intermediate_is_dram = intermediate_type == tt::tt_metal::BufferType::DRAM;
    constexpr bool output_is_dram = output_type == tt::tt_metal::BufferType::DRAM;
    InterleavedAddrGenFast<intermediate_is_dram> intermediate_addrgens[num_inputs];
    InterleavedAddrGenFast<output_is_dram> output_addrgens[num_inputs];
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        auto intermediate_addrgen = InterleavedAddrGenFast<intermediate_is_dram>{
            .bank_base_address = intermediate_addresses[input_idx],
            .page_size = intermediate_page_size,
            .data_format = get_dataformat(cb_forward_id)};
        intermediate_addrgens[input_idx] = intermediate_addrgen;
        auto output_addrgen = InterleavedAddrGenFast<output_is_dram>{
            .bank_base_address = output_addresses[input_idx],
            .page_size = intermediate_page_size,
            .data_format = get_dataformat(cb_forward_id)};
        output_addrgens[input_idx] = output_addrgen;
    }
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }

    uint32_t forward_writes = 0;
    uint32_t backward_writes = 0;

    uint32_t row_offset = 0;
    uint32_t tile_id_start = my_chip_id * input_tensor_Wt;
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        // Write out the local slice to both DRAM and forward and backward
        if (gather_dim == 3) {
            tile_id_start = my_chip_id * input_tensor_Wt;
        } else {
            tile_id_start = my_chip_id * input_tensor_Ht * input_tensor_Wt;
        }
        for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
            for (uint32_t row_idx = 0; row_idx < input_tensor_Ht; row_idx++) {
                for (uint32_t col_idx = 0; col_idx < input_tensor_Wt; col_idx += packet_size_in_pages) {
                    cb_wait_front(cb_forward_id, packet_size_in_pages);
                    size_t l1_read_addr = get_read_ptr(cb_forward_id);

                    for (uint32_t j = 0; j < packet_size_in_pages; j += contig_pages_advanced) {
                        uint64_t noc0_dest_noc_addr = get_noc_addr(
                            tile_id_start + row_offset + col_idx + j,
                            output_addrgens[input_idx],
                            0 /*offset*/,
                            0 /*noc_id*/);
                        uint64_t remote_noc0_dest_noc_addr = get_noc_addr(
                            tile_id_start + row_offset + col_idx + j,
                            intermediate_addrgens[input_idx],
                            0 /*offset*/,
                            0 /*noc_id*/);

                        write_and_advance_local_read_address_for_fabric_write(
                            noc0_dest_noc_addr,
                            remote_noc0_dest_noc_addr,
                            pkt_hdr_forward,
                            pkt_hdr_backward,
                            fabric_connection,
                            l1_read_addr,
                            contig_pages_advanced * intermediate_page_size);
                    }
                    cb_pop_front(cb_forward_id, packet_size_in_pages);
                }
                row_offset += output_tensor_Wt;
            }
            row_offset = 0;
            tile_id_start += output_tensor_Wt * output_tensor_Ht;
        }
    }

    // 2. unicast output ready semaphore forward
    uint64_t out_ready_sem_noc_addr_in_pkt_forward =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_backward, 0);
    auto* pkt_hdr_fwd = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc_forward);
    pkt_hdr_fwd->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt_forward,
        static_cast<uint16_t>(1),  // increment 1
        32});
    // Write the unicast packet (forward)
    if (fabric_connection.has_forward_connection()) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        pkt_hdr_fwd->to_chip_unicast(1);
        fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc_forward, sizeof(PACKET_HEADER_TYPE));
    }
    // 2. unicast output ready semaphore backward
    uint64_t out_ready_sem_noc_addr_in_pkt_backward =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_forward, 0);
    auto* pkt_hdr_bwd = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc_backward);
    pkt_hdr_bwd->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt_backward,
        static_cast<uint16_t>(1),  // increment 1
        32});
    // Write the mcast packet (backward)
    if (fabric_connection.has_backward_connection()) {
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        pkt_hdr_bwd->to_chip_unicast(1);
        fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc_backward, sizeof(PACKET_HEADER_TYPE));
    }

    // increment locally
    if (fuse_op) {
        // Synchronize and signal that the local tensor slice is available
        op_signaler_sender.synchronize_workers_and_signal_op(my_chip_id);
    }

    uint32_t forward_writes_expected, backward_writes_expected;
    if (topology == Topology::Linear) {
        forward_writes_expected = num_targets_backward_direction;
        backward_writes_expected = num_targets_forward_direction;
    } else if (topology == Topology::Ring) {
        forward_writes_expected = num_targets_forward_direction - 1;
        backward_writes_expected = num_targets_backward_direction - 1;
    }

    while (((backward_writes < backward_writes_expected) && fabric_connection.has_backward_connection()) ||
           ((forward_writes < forward_writes_expected) && fabric_connection.has_forward_connection())) {
        // unicast forward
        // Did I get something from my left to send to my right?
        // In the linear case, I expect num_targets_backward_direction slices from the left, and check if I have a
        // neighbor to the right
        // In the ring case, I expect to write to the right num_forward_target times
        if ((forward_writes < forward_writes_expected) && fabric_connection.has_forward_connection()) {
            for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
                row_offset = 0;
                int slice_chip_id = my_chip_id - forward_writes - 1;
                uint32_t actual_slice_chip_id = (slice_chip_id < 0) ? ring_size + slice_chip_id : slice_chip_id;
                if (gather_dim == 3) {
                    tile_id_start = actual_slice_chip_id * input_tensor_Wt;
                } else {
                    tile_id_start = actual_slice_chip_id * input_tensor_Ht * input_tensor_Wt;
                }
                for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
                    for (uint32_t row_idx = 0; row_idx < input_tensor_Ht; row_idx++) {
                        for (uint32_t col_idx = 0; col_idx < input_tensor_Wt; col_idx += packet_size_in_pages) {
                            cb_wait_front(cb_forward_id, packet_size_in_pages);
                            size_t l1_read_addr = get_read_ptr(cb_forward_id);
                            for (uint32_t j = 0; j < packet_size_in_pages; j += contig_pages_advanced) {
                                uint64_t remote_noc0_dest_noc_addr = get_noc_addr(
                                    tile_id_start + row_offset + col_idx + j,
                                    intermediate_addrgens[input_idx],
                                    0 /*offset*/,
                                    0 /*noc_id*/);

                                write_and_advance_local_read_address_for_fabric_write_forward(
                                    remote_noc0_dest_noc_addr,
                                    pkt_hdr_forward,
                                    fabric_connection,
                                    l1_read_addr,
                                    contig_pages_advanced * intermediate_page_size);
                            }
                            cb_pop_front(cb_forward_id, packet_size_in_pages);
                        }
                        row_offset += output_tensor_Wt;
                    }
                    row_offset = 0;
                    tile_id_start += output_tensor_Wt * output_tensor_Ht;
                }
            }
            // 2. unicast output ready semaphore forward
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            pkt_hdr_fwd->to_chip_unicast(1);
            fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc_forward, sizeof(PACKET_HEADER_TYPE));

            forward_writes++;
        }

        // unicast backward
        // Did I get something from my right to send to my left?
        // In the linear case, I expect num_targets_forward_direction slices from the right, and check if I have a
        // neighbor to the left
        // In the ring case, I expect to write to the left num_backward_target times
        if ((backward_writes < backward_writes_expected) && fabric_connection.has_backward_connection()) {
            for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
                row_offset = 0;
                uint32_t slice_chip_id = my_chip_id + backward_writes + 1;
                uint32_t actual_slice_chip_id =
                    (slice_chip_id >= ring_size) ? slice_chip_id - ring_size : slice_chip_id;
                if (gather_dim == 3) {
                    tile_id_start = actual_slice_chip_id * input_tensor_Wt;
                } else {
                    tile_id_start = actual_slice_chip_id * input_tensor_Ht * input_tensor_Wt;
                }
                for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
                    for (uint32_t row_idx = 0; row_idx < input_tensor_Ht; row_idx++) {
                        for (uint32_t col_idx = 0; col_idx < input_tensor_Wt; col_idx += packet_size_in_pages) {
                            cb_wait_front(cb_backward_id, packet_size_in_pages);
                            size_t l1_read_addr = get_read_ptr(cb_backward_id);
                            for (uint32_t j = 0; j < packet_size_in_pages; j += contig_pages_advanced) {
                                uint64_t remote_noc0_dest_noc_addr = get_noc_addr(
                                    tile_id_start + row_offset + col_idx + j,
                                    intermediate_addrgens[input_idx],
                                    0 /*offset*/,
                                    0 /*noc_id*/);

                                write_and_advance_local_read_address_for_fabric_write_backward(
                                    remote_noc0_dest_noc_addr,
                                    pkt_hdr_backward,
                                    fabric_connection,
                                    l1_read_addr,
                                    contig_pages_advanced * intermediate_page_size);
                            }
                            cb_pop_front(cb_backward_id, packet_size_in_pages);
                        }
                        row_offset += output_tensor_Wt;
                    }
                    row_offset = 0;
                    tile_id_start += output_tensor_Wt * output_tensor_Ht;
                }
            }
            // 2. unicast output ready semaphore backward
            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            pkt_hdr_bwd->to_chip_unicast(1);
            fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc_backward, sizeof(PACKET_HEADER_TYPE));

            backward_writes++;
        }
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();
}
