// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/noc_addr.h"
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

constexpr uint32_t N_DRAM_BANKS = 12;
constexpr uint32_t my_chip_id_x = my_chip_id % N_DRAM_BANKS;
constexpr uint32_t my_chip_id_y = my_chip_id / N_DRAM_BANKS;
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    address_t intermediate_address = get_arg_val<address_t>(arg_idx++);
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    uint32_t input_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t slice_num_pages = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_forward = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_backward = get_arg_val<uint32_t>(arg_idx++);
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
    auto intermediate_addrgen = InterleavedAddrGenFast<intermediate_is_dram>{
        .bank_base_address = intermediate_address,
        .page_size = intermediate_page_size,
        .data_format = get_dataformat(cb_forward_id)};
    constexpr bool output_is_dram = output_type == tt::tt_metal::BufferType::DRAM;
    auto output_addrgen = InterleavedAddrGenFast<output_is_dram>{
        .bank_base_address = output_address,
        .page_size = intermediate_page_size,
        .data_format = get_dataformat(cb_forward_id)};

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }

    uint32_t forward_writes = 0;
    uint32_t backward_writes = 0;

    // Write out the local slice to both DRAM and forward and backward
    uint32_t pages_read_in_row = 0;
    uint32_t row_offset = 0;
    uint32_t tiles_read = 0;
    uint32_t tiles_to_read = slice_num_pages;
    uint32_t tile_id_start = my_chip_id * input_tensor_Wt;
    uint32_t packet_id = 0;
    uint32_t intermediate_packet_id_x = my_chip_id_x;
    uint32_t intermediate_packet_id_y = my_chip_id_y;
    while (tiles_read < tiles_to_read) {
        uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
        cb_wait_front(cb_forward_id, num_pages_to_read);
        size_t l1_read_addr = get_read_ptr(cb_forward_id);

        for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
            uint64_t noc0_dest_noc_addr_first_tile = get_noc_addr(
                tile_id_start + row_offset + pages_read_in_row, output_addrgen, 0 /*offset*/, 0 /*noc_id*/);
            pages_read_in_row += 1;
            if (pages_read_in_row >= input_tensor_Wt) {
                row_offset += output_tensor_Wt;
                pages_read_in_row = 0;
            }

            uint64_t noc0_dest_noc_addr_second_tile = get_noc_addr(
                tile_id_start + row_offset + pages_read_in_row, output_addrgen, 0 /*offset*/, 0 /*noc_id*/);
            pages_read_in_row += 1;
            if (pages_read_in_row >= input_tensor_Wt) {
                row_offset += output_tensor_Wt;
                pages_read_in_row = 0;
            }
            uint32_t intermediate_packet_first_tile_id =
                intermediate_packet_id_x + contig_pages_advanced * N_DRAM_BANKS * intermediate_packet_id_y;
            uint64_t remote_noc0_dest_noc_addr =
                get_noc_addr(intermediate_packet_first_tile_id, intermediate_addrgen, 0 /*offset*/, 0 /*noc_id*/);

            write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr_first_tile,
                noc0_dest_noc_addr_second_tile,
                remote_noc0_dest_noc_addr,
                pkt_hdr_forward,
                pkt_hdr_backward,
                fabric_connection,
                l1_read_addr,
                intermediate_page_size,
                contig_pages_advanced);
            tiles_read += contig_pages_advanced;
            packet_id++;

            intermediate_packet_id_x += ring_size;
            if (intermediate_packet_id_x >= N_DRAM_BANKS) {
                intermediate_packet_id_x -= N_DRAM_BANKS;
                intermediate_packet_id_y++;
            }
        }
        cb_pop_front(cb_forward_id, num_pages_to_read);
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

    uint32_t actual_backward_slice_id_x = my_chip_id_x;
    uint32_t actual_backward_slice_id_y = my_chip_id_y;
    uint32_t actual_forward_slice_id_x = my_chip_id_x;
    uint32_t actual_forward_slice_id_y = my_chip_id_y;
    while (((backward_writes < backward_writes_expected) && fabric_connection.has_backward_connection()) ||
           ((forward_writes < forward_writes_expected) && fabric_connection.has_forward_connection())) {
        // unicast forward
        // Did I get something from my left to send to my right?
        // In the linear case, I expect num_targets_backward_direction slices from the left, and check if I have a
        // neighbor to the right
        // In the ring case, I expect to write to the right num_forward_target times

        if ((forward_writes < forward_writes_expected) && fabric_connection.has_forward_connection()) {
            tiles_read = 0;
            tiles_to_read = slice_num_pages;
            actual_forward_slice_id_x =
                (actual_forward_slice_id_x == 0) ? ring_size - 1 : actual_forward_slice_id_x - 1;

            uint32_t packet_id = 0;

            intermediate_packet_id_x = actual_forward_slice_id_x;
            intermediate_packet_id_y = actual_forward_slice_id_y;
            while (tiles_read < tiles_to_read) {
                uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                cb_wait_front(cb_forward_id, num_pages_to_read);
                size_t l1_read_addr = get_read_ptr(cb_forward_id);
                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    uint32_t intermediate_packet_first_tile_id =
                        intermediate_packet_id_x + contig_pages_advanced * N_DRAM_BANKS * intermediate_packet_id_y;
                    uint64_t remote_noc0_dest_noc_addr = get_noc_addr(
                        intermediate_packet_first_tile_id, intermediate_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                    write_and_advance_local_read_address_for_fabric_write_forward(
                        remote_noc0_dest_noc_addr,
                        pkt_hdr_forward,
                        fabric_connection,
                        l1_read_addr,
                        contig_pages_advanced * intermediate_page_size);

                    tiles_read += contig_pages_advanced;
                    packet_id++;
                    intermediate_packet_id_x += ring_size;
                    if (intermediate_packet_id_x >= N_DRAM_BANKS) {
                        intermediate_packet_id_x -= N_DRAM_BANKS;
                        intermediate_packet_id_y++;
                    }
                }
                cb_pop_front(cb_forward_id, num_pages_to_read);
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
            tiles_read = 0;
            tiles_to_read = slice_num_pages;
            actual_backward_slice_id_x =
                (actual_backward_slice_id_x == ring_size - 1) ? 0 : actual_backward_slice_id_x + 1;

            uint32_t packet_id = 0;
            intermediate_packet_id_x = actual_backward_slice_id_x;
            intermediate_packet_id_y = actual_backward_slice_id_y;
            while (tiles_read < tiles_to_read) {
                uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                cb_wait_front(cb_backward_id, num_pages_to_read);
                size_t l1_read_addr = get_read_ptr(cb_backward_id);
                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    uint32_t intermediate_packet_first_tile_id =
                        intermediate_packet_id_x + contig_pages_advanced * N_DRAM_BANKS * intermediate_packet_id_y;
                    uint64_t remote_noc0_dest_noc_addr = get_noc_addr(
                        intermediate_packet_first_tile_id, intermediate_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                    write_and_advance_local_read_address_for_fabric_write_backward(
                        remote_noc0_dest_noc_addr,
                        pkt_hdr_backward,
                        fabric_connection,
                        l1_read_addr,
                        contig_pages_advanced * intermediate_page_size);

                    tiles_read += contig_pages_advanced;
                    packet_id++;
                    intermediate_packet_id_x += ring_size;
                    if (intermediate_packet_id_x >= N_DRAM_BANKS) {
                        intermediate_packet_id_x -= N_DRAM_BANKS;
                        intermediate_packet_id_y++;
                    }
                }
                cb_pop_front(cb_backward_id, num_pages_to_read);
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
