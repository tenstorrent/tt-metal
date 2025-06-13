// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
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
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);  // 4
constexpr BufferType intermediate_type = static_cast<BufferType>(get_compile_time_arg_val(3));
constexpr BufferType output_type = static_cast<BufferType>(get_compile_time_arg_val(4));
constexpr uint32_t cb_compute_output_id = get_compile_time_arg_val(5);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(6);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(7);
constexpr uint32_t intermediate_page_size = get_compile_time_arg_val(8);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(9);
constexpr uint32_t batch_slice_num_pages = get_compile_time_arg_val(10);
constexpr uint32_t ring_size = get_compile_time_arg_val(11);
constexpr uint32_t num_batches = get_compile_time_arg_val(12);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    address_t intermediate_address = get_arg_val<address_t>(arg_idx++);
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_fwd = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_bwd = get_arg_val<uint32_t>(arg_idx++);
    size_t batch_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);

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
    pkt_hdr_forward->to_chip_unicast(1);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
    pkt_hdr_backward->to_chip_unicast(1);

    constexpr uint32_t slice_Wt = input_tensor_Wt / ring_size;

    uint32_t contig_pages_advanced = 1;  // always 1 for interleaved
    uint32_t payload_size_bytes = contig_pages_advanced * intermediate_page_size;

    // interleaved addrgen
    constexpr bool intermediate_is_dram = intermediate_type == tt::tt_metal::BufferType::DRAM;
    auto intermediate_addrgen = InterleavedAddrGenFast<intermediate_is_dram>{
        .bank_base_address = intermediate_address,
        .page_size = intermediate_page_size,
        .data_format = get_dataformat(cb_compute_output_id)};
    constexpr bool output_is_dram = output_type == tt::tt_metal::BufferType::DRAM;
    auto output_addrgen = InterleavedAddrGenFast<output_is_dram>{
        .bank_base_address = output_address,
        .page_size = intermediate_page_size,
        .data_format = get_dataformat(cb_compute_output_id)};

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }

    for (uint32_t b = 0; b < num_batches; b++) {
        uint32_t actual_fwd_slice_idx = my_chip_id;
        uint32_t actual_bwd_slice_idx = my_chip_id;

        uint32_t batch_offset = batch_slice_num_pages * b;
        for (uint32_t i = 0; i < ring_size; ++i) {
            // Next slice idx
            actual_fwd_slice_idx = (actual_fwd_slice_idx == 0) ? ring_size - 1 : actual_fwd_slice_idx - 1;
            actual_bwd_slice_idx = (actual_bwd_slice_idx == ring_size - 1) ? 0 : actual_bwd_slice_idx + 1;

            uint32_t cb_output_id = i > 0 ? cb_compute_output_id : cb_reader_output_id;
            // If not the last slice, write what's on cb_output_id forward
            if (i < (ring_size - 1)) {
                uint32_t pages_read_in_row = 0;
                uint32_t row_offset = 0;
                uint32_t tiles_read = 0;
                uint32_t tiles_to_read = batch_slice_num_pages;
                uint32_t fwd_tile_id_start = actual_fwd_slice_idx * slice_Wt;
                uint32_t bwd_tile_id_start = actual_bwd_slice_idx * slice_Wt;
                bool write_forward = true;
                while (tiles_read < tiles_to_read) {
                    // Alternate writes in forward and backward direction
                    cb_wait_front(cb_output_id, tile_granularity);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);
                    uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);

                    for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                        if (write_forward) {
                            uint64_t remote_noc0_dest_noc_addr = get_noc_addr(
                                fwd_tile_id_start + row_offset + pages_read_in_row,
                                intermediate_addrgen,
                                0 /*offset*/,
                                0 /*noc_id*/);
                            pkt_hdr_forward->to_noc_unicast_write(
                                tt::tt_fabric::NocUnicastCommandHeader{remote_noc0_dest_noc_addr}, payload_size_bytes);
                            if (fabric_connection.has_forward_connection()) {
                                fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                                fabric_connection.get_forward_connection()
                                    .send_payload_without_header_non_blocking_from_address(
                                        l1_read_addr, payload_size_bytes);
                                fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
                                    (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
                            }
                        } else {
                            uint64_t remote_noc0_dest_noc_addr = get_noc_addr(
                                bwd_tile_id_start + row_offset + pages_read_in_row,
                                intermediate_addrgen,
                                0 /*offset*/,
                                0 /*noc_id*/);
                            pkt_hdr_backward->to_noc_unicast_write(
                                tt::tt_fabric::NocUnicastCommandHeader{remote_noc0_dest_noc_addr}, payload_size_bytes);
                            if (fabric_connection.has_backward_connection()) {
                                fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                                fabric_connection.get_backward_connection()
                                    .send_payload_without_header_non_blocking_from_address(
                                        l1_read_addr, payload_size_bytes);
                                fabric_connection.get_backward_connection()
                                    .send_payload_flush_non_blocking_from_address(
                                        (uint32_t)pkt_hdr_backward, sizeof(PACKET_HEADER_TYPE));
                            }
                        }
                        // Note: Must flush write for correctness
                        noc_async_writes_flushed();

                        l1_read_addr += payload_size_bytes;

                        tiles_read += contig_pages_advanced;
                        pages_read_in_row += contig_pages_advanced;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += input_tensor_Wt;
                            pages_read_in_row = 0;
                        }
                    }

                    cb_pop_front(cb_output_id, tile_granularity);
                    write_forward = !write_forward;
                }

                // 2. unicast output ready semaphore forward
                uint64_t out_ready_sem_noc_addr_in_pkt_forward =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_fwd, 0);
                auto* pkt_hdr_fwd = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc_forward);
                pkt_hdr_fwd->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    out_ready_sem_noc_addr_in_pkt_forward,
                    static_cast<uint16_t>(1),  // increment 1
                    32});
                // Write the unicast packet (forward)
                if (fabric_connection.has_forward_connection()) {
                    pkt_hdr_fwd->to_chip_unicast(1);
                    fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                    fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                        packet_header_buffer_seminc_forward, sizeof(PACKET_HEADER_TYPE));
                }
                noc_async_writes_flushed();
                // 2. unicast output ready semaphore backward
                uint64_t out_ready_sem_noc_addr_in_pkt_backward =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bwd, 0);
                auto* pkt_hdr_bwd = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc_backward);
                pkt_hdr_bwd->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    out_ready_sem_noc_addr_in_pkt_backward,
                    static_cast<uint16_t>(1),  // increment 1
                    32});
                // Write the unicast packet (backward)
                if (fabric_connection.has_backward_connection()) {
                    pkt_hdr_bwd->to_chip_unicast(1);
                    fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                    fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                        packet_header_buffer_seminc_backward, sizeof(PACKET_HEADER_TYPE));
                }
                noc_async_writes_flushed();
            } else {
                // Otherwise, on the last slice, write it to output buffer
                uint32_t tiles_read = 0;
                uint32_t tiles_to_read = batch_slice_num_pages;
                uint32_t tile_id_start = batch_offset;
                while (tiles_read < tiles_to_read) {
                    cb_wait_front(cb_output_id, tile_granularity);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);
                    uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);

                    for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                        for (uint32_t t = 0; t < contig_pages_advanced; t++) {
                            noc_async_write_tile(tile_id_start + tiles_read, output_addrgen, l1_read_addr);
                            l1_read_addr += intermediate_page_size;
                            tiles_read++;
                        }
                    }

                    noc_async_writes_flushed();
                    cb_pop_front(cb_output_id, tile_granularity);
                }
                noc_async_write_barrier();

                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_fwd) = 0;
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bwd) = 0;

                // 2. mcast batch ready semaphore forward
                uint64_t out_ready_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, batch_ready_sem, 0);
                auto* pkt_hdr_fwd = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc_forward);
                pkt_hdr_fwd->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    out_ready_sem_noc_addr_in_pkt,
                    static_cast<uint16_t>(1),  // increment 1
                    32});
                // Write the mcast packet (forward)
                if (fabric_connection.has_forward_connection()) {
                    fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                    pkt_hdr_fwd->to_chip_multicast(
                        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(ring_size - 1)});
                    fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                        packet_header_buffer_seminc_forward, sizeof(PACKET_HEADER_TYPE));
                }
                noc_async_writes_flushed();
            }
        }
        // Reset the global semaphore before the next batch
        while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem) < ring_size - 1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem) = 0;
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();
}
