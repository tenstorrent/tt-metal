// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
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
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(13);
constexpr bool direction = get_compile_time_arg_val(14);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    address_t intermediate_address = get_arg_val<address_t>(arg_idx++);
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t batch_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t link = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_links = get_arg_val<uint32_t>(arg_idx++);

    uint32_t slice_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);
    int32_t start_tiles_read = get_arg_val<int32_t>(arg_idx++);
    uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);

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
        int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;

        uint32_t batch_slice_offset = batch_slice_num_pages * b;
        for (uint32_t i = 0; i < ring_size; ++i) {
            uint32_t actual_slice_idx;
            if (direction) {
                actual_slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
            } else {
                actual_slice_idx = slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
            }

            uint32_t cb_output_id = i > 0 ? cb_compute_output_id : cb_reader_output_id;
            // If not the last slice, write what's on cb_output_id forward
            if (i < (ring_size - 1)) {
                uint32_t stride_Wt = input_tensor_Wt;
                uint32_t pages_read_in_row = start_pages_read_in_row;
                uint32_t row_offset = start_row_offset;
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;
                uint32_t input_tile_id_start = actual_slice_idx * slice_Wt;
                if (!direction) {
                    uint32_t backwards_offset = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                    tiles_read += backwards_offset;
                    pages_read_in_row += backwards_offset;

                    if (pages_read_in_row >= slice_Wt) {
                        row_offset += stride_Wt;
                        pages_read_in_row = pages_read_in_row - slice_Wt;
                    }
                }

                while (tiles_read < tiles_to_read) {
                    uint32_t num_pages_to_read = 0;
                    if (direction) {
                        num_pages_to_read = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                    } else {
                        num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);
                    }
                    cb_wait_front(cb_output_id, tile_granularity);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);

                    for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                        uint32_t payload_size_bytes =
                            std::min(contig_pages_advanced, num_pages_to_read - j) * intermediate_page_size;
                        if (direction) {
                            uint64_t remote_noc0_dest_noc_addr = get_noc_addr(
                                input_tile_id_start + row_offset + pages_read_in_row,
                                intermediate_addrgen,
                                0 /*offset*/,
                                0 /*noc_id*/);
                            pkt_hdr->to_noc_unicast_write(
                                tt::tt_fabric::NocUnicastCommandHeader{remote_noc0_dest_noc_addr}, payload_size_bytes);
                            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                            fabric_connection.get_forward_connection()
                                .send_payload_without_header_non_blocking_from_address(
                                    l1_read_addr, payload_size_bytes);
                            fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
                                (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
                        } else {
                            uint64_t remote_noc0_dest_noc_addr = get_noc_addr(
                                input_tile_id_start + row_offset + pages_read_in_row,
                                intermediate_addrgen,
                                0 /*offset*/,
                                0 /*noc_id*/);
                            pkt_hdr->to_noc_unicast_write(
                                tt::tt_fabric::NocUnicastCommandHeader{remote_noc0_dest_noc_addr}, payload_size_bytes);
                            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                            fabric_connection.get_backward_connection()
                                .send_payload_without_header_non_blocking_from_address(
                                    l1_read_addr, payload_size_bytes);
                            fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
                                (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
                        }
                        // Note: Must flush write for correctness
                        noc_async_writes_flushed();
                        l1_read_addr += payload_size_bytes;
                        tiles_read++;

                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }
                    }

                    cb_pop_front(cb_output_id, tile_granularity);

                    // Skip the tiles going the other direction
                    if (tiles_read < tiles_to_read) {
                        num_pages_to_read = 0;
                        if (!direction) {
                            num_pages_to_read = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                        } else {
                            num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);
                        }
                        tiles_read += num_pages_to_read;
                        pages_read_in_row += num_pages_to_read;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = pages_read_in_row - slice_Wt;
                        }
                    }
                }

                // 2. unicast output ready semaphore
                uint64_t out_ready_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
                auto* pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
                pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    out_ready_sem_noc_addr_in_pkt,
                    static_cast<uint16_t>(1),  // increment 1
                    32});
                // Write the unicast packet (forward)
                pkt_hdr->to_chip_unicast(1);
                if (direction) {
                    fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                    fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                        packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
                } else {
                    fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                    fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                        packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
                }
                noc_async_writes_flushed();
            } else {
                // Otherwise, on the last slice, write it to output buffer
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;
                uint32_t tile_id_start = batch_slice_offset;
                if (!direction) {
                    tiles_read += std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                }
                while (tiles_read < tiles_to_read) {
                    uint32_t num_pages_to_read = 0;
                    if (direction) {
                        num_pages_to_read = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                    } else {
                        num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);
                    }
                    cb_wait_front(cb_output_id, tile_granularity);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);

                    for (uint32_t j = 0; j < num_pages_to_read; j++) {
                        noc_async_write_tile(tile_id_start + tiles_read, output_addrgen, l1_read_addr);
                        l1_read_addr += intermediate_page_size;
                        tiles_read++;
                    }

                    noc_async_writes_flushed();
                    cb_pop_front(cb_output_id, tile_granularity);

                    // Skip the tiles going the other direction
                    if (tiles_read < tiles_to_read) {
                        num_pages_to_read = 0;
                        if (!direction) {
                            num_pages_to_read = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                        } else {
                            num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);
                        }
                        tiles_read += num_pages_to_read;
                    }
                }
                noc_async_write_barrier();

                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem) = 0;

                // 2. mcast half batch ready semaphore
                uint64_t out_ready_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, batch_ready_sem, 0);
                auto* pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
                pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    out_ready_sem_noc_addr_in_pkt,
                    static_cast<uint16_t>(1),  // increment 1
                    32});
                // Write the mcast packet
                if (direction) {
                    fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                    pkt_hdr->to_chip_multicast(
                        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(ring_size - 1)});
                    fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                        packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
                    noc_async_writes_flushed();
                } else {
                    fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                    pkt_hdr->to_chip_multicast(
                        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(ring_size - 1)});
                    fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                        packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
                    noc_async_writes_flushed();
                }
            }

            // Next slice idx
            if (direction) {
                slice_idx--;
            } else {
                slice_idx++;
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
