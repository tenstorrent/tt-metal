// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
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
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(13);
constexpr bool is_forward = get_compile_time_arg_val(14);
constexpr bool is_first_device_in_direction = get_compile_time_arg_val(15);
constexpr uint32_t num_targets_in_direction = get_compile_time_arg_val(16);
constexpr uint32_t num_intermediate_reduction_steps = get_compile_time_arg_val(17);
constexpr bool do_final_reduction = get_compile_time_arg_val(18);
constexpr uint32_t num_total_reduction_steps = get_compile_time_arg_val(19);
constexpr bool sync_with_other_direction = get_compile_time_arg_val(20);

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
    size_t final_reduction_slot_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t batch_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t link = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_links = get_arg_val<uint32_t>(arg_idx++);
    uint32_t fwd_bwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t opposite_core_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t opposite_core_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
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
    pkt_hdr_seminc->to_chip_unicast(1);

    uint32_t slice_Wt = input_tensor_Wt / ring_size;

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

    auto* dir_fabric_connection =
        fabric_connection.is_logically_connected()
            ? (is_forward ? &fabric_connection.get_forward_connection() : &fabric_connection.get_backward_connection())
            : nullptr;  // Null connection if not connected

    for (uint32_t b = 0; b < num_batches; b++) {
        int slice_idx = is_forward ? ring_size - 1 : 0;

        uint32_t batch_slice_offset = batch_slice_num_pages * b;
        for (uint32_t iter = 0; iter < num_targets_in_direction; ++iter) {
            // Last send is special for backwards - send to different slice idx to avoid overlap
            if constexpr (!is_forward) {
                if (iter == num_targets_in_direction - 1) {
                    // Wait for final_reduction_slot_sem to be signaled
                    // Send to different slice idx to avoid overlap
                    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(final_reduction_slot_sem), 1);
                    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(final_reduction_slot_sem), 0);
                    constexpr bool their_first_write_was_fwd = (my_chip_id - 1) < ring_size / 2;
                    // If their first write was forward, they freed up the last slot first. Otherwise, the freed
                    // up the first slot first. That's where I can write to.
                    if constexpr (their_first_write_was_fwd) {
                        slice_idx = ring_size - 1;
                    } else {
                        slice_idx = 0;
                    }
                }
            }

            constexpr uint32_t cb_output_id = is_first_device_in_direction ? cb_reader_output_id : cb_compute_output_id;

            uint32_t stride_Wt = input_tensor_Wt;
            uint32_t pages_read_in_row = (link * batch_slice_num_pages / num_links) % slice_Wt;
            uint32_t row_offset = (link * batch_slice_num_pages / num_links) / slice_Wt * stride_Wt;
            uint32_t tiles_read = (link * batch_slice_num_pages / num_links);
            uint32_t tiles_to_read = (link + 1) * batch_slice_num_pages / num_links;

            uint32_t input_tile_id_start = slice_idx * slice_Wt;

            // Write to remote intermediate buffer
            while (tiles_read < tiles_to_read) {
                uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);
                cb_wait_front(cb_output_id, tile_granularity);
                size_t l1_read_addr = get_read_ptr(cb_output_id);

                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    uint32_t num_pages_to_write = std::min(contig_pages_advanced, num_pages_to_read - j);
                    uint32_t payload_size_bytes = num_pages_to_write * intermediate_page_size;

                    uint32_t first_tile_id = input_tile_id_start + row_offset + pages_read_in_row;
                    uint64_t remote_noc0_dest_noc_addr =
                        get_noc_addr(first_tile_id, intermediate_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                    pages_read_in_row++;
                    if (pages_read_in_row >= slice_Wt) {
                        row_offset += stride_Wt;
                        pages_read_in_row = 0;
                    }

                    if (num_pages_to_write == 2) {
                        uint32_t second_tile_id = input_tile_id_start + row_offset + pages_read_in_row;
                        uint64_t second_remote_noc0_dest_noc_addr =
                            get_noc_addr(second_tile_id, intermediate_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }

                        pkt_hdr->to_noc_unicast_scatter_write(
                            tt::tt_fabric::NocUnicastScatterCommandHeader{
                                remote_noc0_dest_noc_addr,
                                second_remote_noc0_dest_noc_addr,
                                intermediate_page_size /*single packet size*/},
                            payload_size_bytes /*total payload size*/);
                        dir_fabric_connection->wait_for_empty_write_slot();
                        dir_fabric_connection->send_payload_without_header_non_blocking_from_address(
                            l1_read_addr, payload_size_bytes);
                        dir_fabric_connection->send_payload_flush_non_blocking_from_address(
                            (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));

                    } else {
                        ASSERT(num_pages_to_write == 1);
                        pkt_hdr->to_noc_unicast_write(
                            tt::tt_fabric::NocUnicastCommandHeader{remote_noc0_dest_noc_addr}, payload_size_bytes);
                        dir_fabric_connection->wait_for_empty_write_slot();
                        dir_fabric_connection->send_payload_without_header_non_blocking_from_address(
                            l1_read_addr, payload_size_bytes);
                        dir_fabric_connection->send_payload_flush_non_blocking_from_address(
                            (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
                    }

                    // Note: Must flush write for correctness
                    noc_async_writes_flushed();
                    l1_read_addr += payload_size_bytes;
                    tiles_read += num_pages_to_write;
                }
                cb_pop_front(cb_output_id, tile_granularity);
            }

            // 2. unicast output ready semaphore
            uint64_t out_ready_sem_noc_addr_in_pkt =
                safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
            pkt_hdr_seminc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                out_ready_sem_noc_addr_in_pkt,
                static_cast<uint16_t>(1),  // increment 1
                32});
            pkt_hdr_seminc->to_chip_unicast(1);

            dir_fabric_connection->wait_for_empty_write_slot();
            dir_fabric_connection->send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
            noc_async_writes_flushed();

            if constexpr (is_forward) {
                if (iter == 0) {
                    // First send is special for forwards (tell backwards direction that slot is free)

                    // Signal final_reduction_slot_sem on remote BWD core
                    uint64_t final_slot_sem_noc_addr_in_pkt = safe_get_noc_addr(
                        opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, final_reduction_slot_sem, 0);
                    pkt_hdr_seminc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                        final_slot_sem_noc_addr_in_pkt,
                        static_cast<uint16_t>(1),  // increment 1
                        32});

                    dir_fabric_connection->wait_for_empty_write_slot();
                    dir_fabric_connection->send_payload_flush_blocking_from_address(
                        packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
                    noc_async_writes_flushed();
                }
            }

            // Next slice idx
            if constexpr (is_forward) {
                slice_idx--;
            } else {
                slice_idx++;
            }
        }

        // Do write of final reduction and sync local FWD/BWD cores
        if constexpr (do_final_reduction) {
            // Write output
            uint32_t tiles_read = (link * batch_slice_num_pages / num_links);
            uint32_t tiles_to_read = (link + 1) * batch_slice_num_pages / num_links;
            uint32_t tile_id_start = batch_slice_offset;

            while (tiles_read < tiles_to_read) {
                uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);
                cb_wait_front(cb_compute_output_id, tile_granularity);
                size_t l1_read_addr = get_read_ptr(cb_compute_output_id);

                for (uint32_t j = 0; j < num_pages_to_read; j++) {
                    noc_async_write_tile(tile_id_start + tiles_read, output_addrgen, l1_read_addr);
                    l1_read_addr += intermediate_page_size;
                    tiles_read++;
                }

                noc_async_writes_flushed();
                cb_pop_front(cb_compute_output_id, tile_granularity);
            }

            noc_async_write_barrier();
            if constexpr (sync_with_other_direction && is_forward) {
                // Tell local backwards reader that it can proceed
                uint64_t fwd_bwd_sem_noc_addr =
                    safe_get_noc_addr(opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, fwd_bwd_sem_addr, 0);
                noc_semaphore_inc(fwd_bwd_sem_noc_addr, 1);
            }
        }

        /**
         * Since FWD signals BWD to continue, FWD is ahead of BWD. Make FWD wait for BWD before
         * doing sync on batch.
         * Local FWD signals remote BWD cores that local BWD has consumed its intermediate, so they can proceed.
         * Local BWD signals remote FWD cores that local FWD has consumed its intermediate, so they can proceed.
         */
        // Have local FWD wait on local BWD reaching here
        if constexpr (!is_forward) {
            // Have local BWD tell local FWD that it's done
            uint64_t fwd_bwd_sem_noc_addr =
                safe_get_noc_addr(opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, fwd_bwd_sem_addr, 0);
            noc_semaphore_inc(fwd_bwd_sem_noc_addr, 1);
        } else {
            // Local FWD waits here until BWD has completed writes
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd_bwd_sem_addr), 1);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd_bwd_sem_addr), 0);
        }

        if (fabric_connection.is_logically_connected()) {
            // mcast batch_ready_sem to opposite core in my direction
            uint64_t batch_ready_sem_noc_addr_in_pkt =
                safe_get_noc_addr(opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, batch_ready_sem, 0);
            pkt_hdr_seminc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                batch_ready_sem_noc_addr_in_pkt,
                static_cast<uint16_t>(1),  // increment 1
                32});
            pkt_hdr_seminc->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_in_direction)});

            dir_fabric_connection->wait_for_empty_write_slot();
            dir_fabric_connection->send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
            noc_async_writes_flushed();
        }

        // Reset the global semaphore before the next batch
        // We're going to get hit by however many cores we're targeting, since the opposite core sends back toward us.
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), num_targets_in_direction);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem), 0);
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
