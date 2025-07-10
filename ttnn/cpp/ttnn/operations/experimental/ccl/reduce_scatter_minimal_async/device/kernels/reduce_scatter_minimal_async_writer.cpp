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
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(13);
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
    int32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
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

    volatile PACKET_HEADER_TYPE* pkt_hdr_seminc =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);

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

    auto* fabric_direction_connection =
        direction ? &fabric_connection.get_forward_connection() : &fabric_connection.get_backward_connection();

    for (uint32_t b = 0; b < num_batches; b++) {
        int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;

        uint32_t batch_slice_offset = batch_slice_num_pages * b;
        for (uint32_t i = 0; i < ring_size; ++i) {
            uint32_t actual_slice_idx;
            if constexpr (direction) {
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
                if constexpr (!direction) {
                    uint32_t backwards_offset = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                    for (uint32_t k = 0; k < backwards_offset; ++k) {
                        pages_read_in_row++;
                        if (pages_read_in_row == slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = pages_read_in_row - slice_Wt;
                        }
                    }
                    tiles_read += backwards_offset;
                }

                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;

                    uint32_t tiles_read_in_current_direction = 0;
                    uint32_t tiles_to_read_in_current_direction = 0;
                    if constexpr (direction) {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                    } else {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read, tile_granularity);
                    }

                    cb_wait_front(cb_output_id, tile_granularity);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);
                    while (tiles_read_in_current_direction < tiles_to_read_in_current_direction) {
                        uint32_t tiles_remaining_to_read_in_current_direction =
                            tiles_to_read_in_current_direction - tiles_read_in_current_direction;
                        uint32_t tiles_to_put_in_current_packet =
                            std::min(tiles_remaining_to_read_in_current_direction, num_tiles_to_write_per_packet);

                        // Will have more cases once scatter-write supports more than 2 distinct addresses
                        switch (tiles_to_put_in_current_packet) {
                            case 2: {
                                uint32_t tile_one_id = input_tile_id_start + row_offset + pages_read_in_row;
                                pages_read_in_row++;
                                if (pages_read_in_row == slice_Wt) {
                                    row_offset += stride_Wt;
                                    pages_read_in_row = 0;
                                }

                                uint32_t tile_two_id = input_tile_id_start + row_offset + pages_read_in_row;
                                pages_read_in_row++;
                                if (pages_read_in_row == slice_Wt) {
                                    row_offset += stride_Wt;
                                    pages_read_in_row = 0;
                                }

                                uint64_t remote_noc0_dest_noc_addr_tile_one =
                                    get_noc_addr(tile_one_id, intermediate_addrgen, 0 /*offset*/, 0 /*noc_id*/);
                                uint64_t remote_noc0_dest_noc_addr_tile_two =
                                    get_noc_addr(tile_two_id, intermediate_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                                scatter_write_and_advance_local_read_address_for_fabric(
                                    remote_noc0_dest_noc_addr_tile_one,
                                    remote_noc0_dest_noc_addr_tile_two,
                                    pkt_hdr,
                                    fabric_direction_connection,
                                    l1_read_addr,
                                    intermediate_page_size,
                                    intermediate_page_size);
                                break;
                            }
                            case 1:
                            default: {
                                uint32_t tile_id = input_tile_id_start + row_offset + pages_read_in_row;
                                pages_read_in_row++;
                                if (pages_read_in_row == slice_Wt) {
                                    row_offset += stride_Wt;
                                    pages_read_in_row = 0;
                                }

                                uint64_t remote_noc0_dest_noc_addr =
                                    get_noc_addr(tile_id, intermediate_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                                write_and_advance_local_read_address_for_fabric(
                                    remote_noc0_dest_noc_addr,
                                    pkt_hdr,
                                    fabric_direction_connection,
                                    l1_read_addr,
                                    intermediate_page_size);
                                break;
                            }
                        }
                        tiles_read += tiles_to_put_in_current_packet;
                        tiles_read_in_current_direction += tiles_to_put_in_current_packet;
                    }
                    cb_pop_front(cb_output_id, tile_granularity);

                    // Skip the tiles going the other direction
                    tiles_remaining_to_read = tiles_to_read - tiles_read;
                    if (tiles_remaining_to_read > 0) {
                        uint32_t tiles_to_read_in_other_direction = 0;
                        if constexpr (!direction) {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                        } else {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read, tile_granularity);
                        }

                        for (uint32_t k = 0; k < tiles_to_read_in_other_direction; ++k) {
                            pages_read_in_row++;
                            if (pages_read_in_row == slice_Wt) {
                                row_offset += stride_Wt;
                                pages_read_in_row = pages_read_in_row - slice_Wt;
                            }
                        }
                        tiles_read += tiles_to_read_in_other_direction;
                    }
                }

                // 2. unicast output ready semaphore
                uint64_t out_ready_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
                pkt_hdr_seminc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    out_ready_sem_noc_addr_in_pkt,
                    static_cast<uint16_t>(1),  // increment 1
                    32});
                // Write the unicast packet (forward)
                pkt_hdr_seminc->to_chip_unicast(1);
                fabric_direction_connection->wait_for_empty_write_slot();
                fabric_direction_connection->send_payload_flush_blocking_from_address(
                    packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
                noc_async_writes_flushed();
            } else {
                // Otherwise, on the last slice, write it to output buffer
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;
                uint32_t tile_id_start = batch_slice_offset;
                if constexpr (!direction) {
                    tiles_read += std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                }
                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;

                    uint32_t tiles_to_read_in_current_direction = 0;
                    if constexpr (direction) {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                    } else {
                        tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read, tile_granularity);
                    }

                    cb_wait_front(cb_output_id, tile_granularity);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);
                    for (uint32_t j = 0; j < tiles_to_read_in_current_direction; ++j) {
                        uint32_t tile_id = tile_id_start + tiles_read;
                        noc_async_write_tile(tile_id, output_addrgen, l1_read_addr);
                        l1_read_addr += intermediate_page_size;
                        tiles_read++;
                    }

                    noc_async_write_barrier();
                    cb_pop_front(cb_output_id, tile_granularity);

                    // Skip the tiles going the other direction
                    tiles_remaining_to_read = tiles_to_read - tiles_read;
                    if (tiles_remaining_to_read > 0) {
                        uint32_t tiles_to_read_in_other_direction = 0;
                        if constexpr (!direction) {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
                        } else {
                            tiles_to_read_in_other_direction = std::min(tiles_remaining_to_read, tile_granularity);
                        }
                        tiles_read += tiles_to_read_in_other_direction;
                    }
                }

                // 2. mcast half batch ready semaphore
                uint64_t batch_ready_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, batch_ready_sem, 0);
                pkt_hdr_seminc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    batch_ready_sem_noc_addr_in_pkt,
                    static_cast<uint16_t>(1),  // increment 1
                    32});
                // Write the mcast packet
                fabric_direction_connection->wait_for_empty_write_slot();
                pkt_hdr_seminc->to_chip_multicast(
                    tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(ring_size - 1)});
                fabric_direction_connection->send_payload_flush_blocking_from_address(
                    packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
                noc_async_writes_flushed();
            }

            // Next slice idx
            if constexpr (direction) {
                slice_idx--;
            } else {
                slice_idx++;
            }
        }
        // Reset the global semaphore before the next batch
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), ring_size - 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), 0);
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();
}
