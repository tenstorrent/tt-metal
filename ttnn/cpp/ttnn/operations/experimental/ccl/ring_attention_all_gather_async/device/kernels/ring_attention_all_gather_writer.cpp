// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
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
constexpr BufferType output_type = static_cast<BufferType>(get_compile_time_arg_val(3));
constexpr uint32_t cb_output_id = get_compile_time_arg_val(4);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(5);
constexpr uint32_t output_page_size = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(7);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(8);
constexpr bool dynamic_alternate = get_compile_time_arg_val(9);
constexpr bool fuse_op = get_compile_time_arg_val(10);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(11));
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(12);
constexpr uint32_t num_inputs = get_compile_time_arg_val(13);
constexpr bool direction = get_compile_time_arg_val(14);  // 1 is forward, 0 is backward

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
    uint32_t input_tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tile_id_end = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    address_t output_addresses[num_inputs];
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        address_t output_address = get_arg_val<address_t>(arg_idx++);
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
    auto packet_header_buffer_addr = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
    pkt_hdr->to_chip_unicast(1);

    // interleaved addrgen
    constexpr bool output_is_dram = output_type == tt::tt_metal::BufferType::DRAM;
    InterleavedAddrGenFast<output_is_dram> output_addrgens[num_inputs];
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        auto output_addrgen = InterleavedAddrGenFast<output_is_dram>{
            .bank_base_address = output_addresses[input_idx],
            .page_size = output_page_size,
            .data_format = get_dataformat(cb_output_id)};
        output_addrgens[input_idx] = output_addrgen;
    }

    fabric_connection.open();

    tt::tt_fabric::WorkerToFabricEdmSender* fabric_direction_connection =
        fabric_connection.is_logically_connected() ? (direction == 1 ? &fabric_connection.get_backward_connection()
                                                                     : &fabric_connection.get_forward_connection())
                                                   : nullptr;
    constexpr uint32_t num_targets_in_direction =
        direction == 1 ? num_targets_backward_direction : num_targets_forward_direction;

    uint32_t slice_writes = 0;

    uint32_t row_offset = 0;
    uint32_t tile_id_start = my_chip_id * input_tensor_Wt;
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        // Write out the local slice to both DRAM and forward and backward
        uint32_t pages_read_in_row = input_tile_id_start % input_tensor_Wt;
        uint32_t row_offset = (input_tile_id_start / input_tensor_Wt) * output_tensor_Wt;
        uint32_t tiles_read = input_tile_id_start;
        uint32_t tiles_to_read = input_tile_id_end;
        uint32_t tile_id_start = my_chip_id * input_tensor_Wt;
        if (gather_dim == 3) {
            tile_id_start = my_chip_id * input_tensor_Wt;
        } else {
            tile_id_start = my_chip_id * input_tensor_Ht * input_tensor_Wt;
        }
        for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
            while (tiles_read < tiles_to_read) {
                uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                cb_wait_front(cb_output_id, packet_size_in_pages);
                const size_t l1_read_addr_base = get_read_ptr(cb_output_id);
                size_t l1_read_addr = l1_read_addr_base;

                // for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                uint32_t tile_id = tile_id_start + row_offset + pages_read_in_row;
                uint64_t noc0_dest_noc_addr =
                    get_noc_addr(tile_id, output_addrgens[input_idx], 0 /*offset*/, 0 /*noc_id*/);

                pages_read_in_row++;
                if (pages_read_in_row >= input_tensor_Wt) {
                    row_offset += output_tensor_Wt;
                    pages_read_in_row = 0;
                }

                if (num_pages_to_read == 2) {
                    uint32_t second_tile_id = tile_id_start + row_offset + pages_read_in_row;
                    uint64_t second_noc0_dest_noc_addr =
                        get_noc_addr(second_tile_id, output_addrgens[input_idx], 0 /*offset*/, 0 /*noc_id*/);

                    if constexpr (direction == 1) {
                        // Backwards does local write
                        noc_async_write_tile(tile_id, output_addrgens[input_idx], l1_read_addr);
                        noc_async_write_tile(
                            second_tile_id, output_addrgens[input_idx], l1_read_addr + output_page_size);
                    }

                    if constexpr (num_targets_in_direction) {
                        scatter_fabric_write_unidir(
                            noc0_dest_noc_addr,
                            second_noc0_dest_noc_addr,
                            pkt_hdr,
                            *fabric_direction_connection,
                            l1_read_addr,
                            (uint16_t)output_page_size,
                            output_page_size);
                    }

                    pages_read_in_row++;
                    if (pages_read_in_row >= input_tensor_Wt) {
                        row_offset += output_tensor_Wt;
                        pages_read_in_row = 0;
                    }
                } else {
                    ASSERT(num_pages_to_read == 1);
                    if constexpr (direction == 1) {
                        noc_async_write_tile(tile_id, output_addrgens[input_idx], l1_read_addr);
                    }
                    if constexpr (num_targets_in_direction) {
                        // Has valid targets to send to
                        fabric_write_unidir(
                            noc0_dest_noc_addr, pkt_hdr, *fabric_direction_connection, l1_read_addr, output_page_size);
                    }
                }

                tiles_read += num_pages_to_read;

                cb_pop_front(cb_output_id, packet_size_in_pages);
            }
            tile_id_start += output_tensor_Wt * output_tensor_Ht;
            tiles_read = input_tile_id_start;
            tiles_to_read = input_tile_id_end;
            pages_read_in_row = input_tile_id_start % input_tensor_Wt;
            row_offset = (input_tile_id_start / input_tensor_Wt) * output_tensor_Wt;
        }
    }

    noc_async_write_barrier();
    // increment locally
    if constexpr (fuse_op && direction == 1) {
        // Synchronize and signal that the local tensor slice is available
        op_signaler_sender.synchronize_workers_and_signal_op(my_chip_id);
    }

    // 2. unicast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
    auto* pkt_hdr_sem_inc = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
    pkt_hdr_sem_inc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt,
        static_cast<uint16_t>(1),  // increment 1
        32});

    // Write the unicast packet
    if constexpr (num_targets_in_direction) {
        fabric_direction_connection->wait_for_empty_write_slot();
        pkt_hdr_sem_inc->to_chip_unicast(1);
        fabric_direction_connection->send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }

    uint32_t writes_expected = 0;
    if constexpr (topology == Topology::Linear) {
        if constexpr (direction == 1 && num_targets_backward_direction) {
            writes_expected = num_targets_forward_direction;
        } else if constexpr (direction == 0 && num_targets_forward_direction) {
            writes_expected = num_targets_backward_direction;
        }
    } else if constexpr (topology == Topology::Ring) {
        if constexpr (direction == 1) {
            writes_expected = num_targets_backward_direction - 1;
        } else {
            writes_expected = num_targets_forward_direction - 1;
        }
    }

    while (slice_writes < writes_expected) {
        // Direction == backward
        // Did I get something from my left to send to my right?
        // In the linear case, I expect num_targets_backward_direction slices from the left, and check if I have a
        // neighbor to the right
        // In the ring case, I expect to write to the right num_forward_target times
        // Direction == forward
        // Did I get something from my right to send to my left?
        // In the linear case, I expect num_targets_forward_direction slices from the right, and check if I have a
        // neighbor to the left
        // In the ring case, I expect to write to the left num_backward_target times

        int slice_chip_id;
        uint32_t actual_slice_chip_id;
        if constexpr (direction == 1) {
            slice_chip_id = my_chip_id + slice_writes + 1;
            actual_slice_chip_id = (slice_chip_id >= (int)ring_size) ? slice_chip_id - ring_size : slice_chip_id;
        } else {
            slice_chip_id = my_chip_id - slice_writes - 1;
            actual_slice_chip_id = (slice_chip_id < 0) ? ring_size + slice_chip_id : slice_chip_id;
        }
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            uint32_t tiles_read = input_tile_id_start;
            uint32_t tiles_to_read = input_tile_id_end;
            uint32_t tile_id_start = actual_slice_chip_id * input_tensor_Wt;
            uint32_t row_offset = (input_tile_id_start / input_tensor_Wt) * output_tensor_Wt;
            uint32_t pages_read_in_row = (input_tile_id_start % input_tensor_Wt);
            uint32_t slice_Wt = input_tensor_Wt;
            uint32_t stride_Wt = output_tensor_Wt;

            if (gather_dim == 3) {
                tile_id_start = actual_slice_chip_id * input_tensor_Wt;
            } else {
                tile_id_start = actual_slice_chip_id * input_tensor_Ht * input_tensor_Wt;
            }
            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
                while (tiles_read < tiles_to_read) {
                    uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                    cb_wait_front(cb_output_id, packet_size_in_pages);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);
                    uint64_t noc0_dest_noc_addr = get_noc_addr(
                        tile_id_start + row_offset + pages_read_in_row,
                        output_addrgens[input_idx],
                        0 /*offset*/,
                        0 /*noc_id*/);
                    pages_read_in_row++;
                    if (pages_read_in_row >= slice_Wt) {
                        row_offset += stride_Wt;
                        pages_read_in_row = 0;
                    }

                    if (num_pages_to_read == 2) {
                        uint64_t second_noc0_dest_noc_addr = get_noc_addr(
                            tile_id_start + row_offset + pages_read_in_row,
                            output_addrgens[input_idx],
                            0 /*offset*/,
                            0 /*noc_id*/);
                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }

                        scatter_fabric_write_unidir(
                            noc0_dest_noc_addr,
                            second_noc0_dest_noc_addr,
                            pkt_hdr,
                            *fabric_direction_connection,
                            l1_read_addr,
                            (uint16_t)output_page_size,
                            output_page_size);
                    } else {
                        ASSERT(num_pages_to_read == 1);
                        fabric_write_unidir(
                            noc0_dest_noc_addr, pkt_hdr, *fabric_direction_connection, l1_read_addr, output_page_size);
                    }

                    tiles_read += num_pages_to_read;
                    cb_pop_front(cb_output_id, packet_size_in_pages);
                }
                tile_id_start += output_tensor_Wt * output_tensor_Ht;
                tiles_read = input_tile_id_start;
                tiles_to_read = input_tile_id_end;
                row_offset = (input_tile_id_start / input_tensor_Wt) * output_tensor_Wt;
                pages_read_in_row = (input_tile_id_start % input_tensor_Wt);
            }
        }

        // 2. unicast output ready semaphore forward
        fabric_direction_connection->wait_for_empty_write_slot();
        pkt_hdr_sem_inc->to_chip_unicast(1);
        fabric_direction_connection->send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));

        slice_writes++;
    }
    fabric_connection.close();

    noc_async_atomic_barrier();
    noc_async_write_barrier();
}
