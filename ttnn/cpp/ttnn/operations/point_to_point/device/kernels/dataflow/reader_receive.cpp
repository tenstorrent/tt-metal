// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "../common.hpp"

using tt::data_movement::common::tt_memmove;

void kernel_main() {
    constexpr bool intermediate_is_dram = get_compile_time_arg_val(0);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t receiver_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t alignment = get_compile_time_arg_val(4);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    const auto page_idx_start = get_arg_val<uint32_t>(0);
    const auto page_idx_end = get_arg_val<uint32_t>(1);
    const auto max_pages_per_packet = get_arg_val<uint32_t>(2);
    const auto intermediate_base_addr = get_arg_val<uint32_t>(3);
    const auto packet_size_bytes = get_arg_val<uint32_t>(4);
    const auto page_size_bytes = get_arg_val<uint32_t>(5);
    const auto page_segments = get_arg_val<uint32_t>(6);
    const uint32_t sender_semaphore_addr = get_arg_val<uint32_t>(7);
    const uint8_t sender_num_hops = get_arg_val<uint32_t>(8);
    const bool sender_is_forward = get_arg_val<uint32_t>(9);

    // reusing the last arg for fabric setup, therefore index overlaps.
    size_t conn_arg_idx = 9;

    auto fabric_connection = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(conn_arg_idx);

    cb_reserve_back(packet_header_cb_id, 1);
    const uint32_t sem_header_addr = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    const uint64_t sender_sem_noc_addr = get_noc_addr(sender_semaphore_addr);

    auto* sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr);
    sem_header_ptr->to_chip_unicast(sender_num_hops);
    sem_header_ptr->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_sem_noc_addr, 1, 32});

    fabric_connection.open_finish();
    auto& connection_direction =
        tt::point_to_point::common::connection_direction_collection(sender_is_forward, fabric_connection);

    connection_direction.wait_for_empty_write_slot();
    connection_direction.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr, packet_header_size_bytes);

    fabric_connection.close();

    InterleavedAddrGen<intermediate_is_dram> packet_buffer_addrgen{
        .bank_base_address = intermediate_base_addr, .page_size = packet_size_bytes};

    cb_reserve_back(packet_cb_id, 1);
    const uint64_t packet_l1_addr = get_write_ptr(packet_cb_id);

    auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
    noc_semaphore_wait(local_semaphore_ptr, 1);

    const uint32_t aligned_page_size_bytes = align(page_size_bytes, alignment);
    uint32_t curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx_start);
    uint32_t packet_idx = page_idx_start / max_pages_per_packet;

    for (uint32_t page_idx = page_idx_start, packet_page_idx = 0; page_idx < page_idx_end; ++page_idx) {
        cb_reserve_back(receiver_cb_id, 1);
        const uint32_t dest_page_base_addr = get_write_ptr(receiver_cb_id);

        for (uint32_t page_segment_idx = 0; page_segment_idx < page_segments; ++page_segment_idx) {
            if (page_idx == page_idx_start || packet_page_idx == curr_pages_per_packet) {
                const uint64_t packet_noc_addr = packet_buffer_addrgen.get_noc_addr(packet_idx);
                noc_async_read(packet_noc_addr, packet_l1_addr, packet_size_bytes);
                noc_async_read_barrier();

                packet_page_idx = 0;
                curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx);
                ++packet_idx;
            }

            const uint32_t page_offset = page_segment_idx * packet_size_bytes;
            const uint32_t dest_addr = dest_page_base_addr + page_offset;
            const uint32_t transfer_size_bytes = std::min(page_size_bytes - page_offset, packet_size_bytes);
            const uint32_t packet_l1_page_addr = packet_l1_addr + packet_page_idx * aligned_page_size_bytes;

            tt_memmove<false, false, false, 0>(dest_addr, packet_l1_page_addr, transfer_size_bytes);
            ++packet_page_idx;
        }
        cb_push_back(receiver_cb_id, 1);
    }
    cb_push_back(packet_cb_id, 1);

    // clean up semaphore in case it is reused
    noc_semaphore_set(local_semaphore_ptr, 0);
}
