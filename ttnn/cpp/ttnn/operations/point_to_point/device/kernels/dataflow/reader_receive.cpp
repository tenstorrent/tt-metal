// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "../common.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using tt::data_movement::common::tt_memmove;

void kernel_main() {
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t receiver_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t alignment = get_compile_time_arg_val(3);
    constexpr auto packet_buffer_args = TensorAccessorArgs<4>();

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

    Noc noc;
    CircularBuffer packet_header_cb(packet_header_cb_id);
    CircularBuffer packet_cb(packet_cb_id);
    CircularBuffer receiver_cb(receiver_cb_id);

    auto fabric_connection = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(conn_arg_idx);

    packet_header_cb.reserve_back(1);
    const uint32_t sem_header_addr = packet_header_cb.get_write_ptr();
    packet_header_cb.push_back(1);

    const uint64_t sender_sem_noc_addr = get_noc_addr(sender_semaphore_addr);

    auto* sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)sem_header_ptr, sender_num_hops);
    sem_header_ptr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_sem_noc_addr, 1});

    fabric_connection.open_finish();
    auto& connection_direction =
        tt::point_to_point::common::connection_direction_collection(sender_is_forward, fabric_connection);

    connection_direction.wait_for_empty_write_slot();
    connection_direction.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr, packet_header_size_bytes);

    fabric_connection.close();

    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    const auto packet_buffer = TensorAccessor(packet_buffer_args, intermediate_base_addr, packet_size_bytes);

    packet_cb.reserve_back(1);
    const uint64_t packet_l1_addr = packet_cb.get_write_ptr();

    auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
    noc_semaphore_wait(local_semaphore_ptr, 1);

    const uint32_t aligned_page_size_bytes = align(page_size_bytes, alignment);
    uint32_t curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx_start);
    uint32_t packet_idx = page_idx_start / max_pages_per_packet;

    for (uint32_t page_idx = page_idx_start, packet_page_idx = 0; page_idx < page_idx_end; ++page_idx) {
        receiver_cb.reserve_back(1);
        const uint32_t dest_page_base_addr = receiver_cb.get_write_ptr();

        for (uint32_t page_segment_idx = 0; page_segment_idx < page_segments; ++page_segment_idx) {
            if (page_idx == page_idx_start || packet_page_idx == curr_pages_per_packet) {
                noc.async_read(
                    packet_buffer, packet_cb, packet_size_bytes, {.page_id = packet_idx}, {.offset_bytes = 0});
                noc.async_read_barrier();

                packet_page_idx = 0;
                curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx);
                ++packet_idx;
            }

            const uint32_t page_offset = page_segment_idx * packet_size_bytes;
            const uint32_t dest_addr = dest_page_base_addr + page_offset;
            const uint32_t transfer_size_bytes = std::min(page_size_bytes - page_offset, packet_size_bytes);
            const uint32_t packet_l1_page_addr = packet_l1_addr + packet_page_idx * aligned_page_size_bytes;

            tt_memmove<false, false, false, 0>(noc, dest_addr, packet_l1_page_addr, transfer_size_bytes);
            ++packet_page_idx;
        }
        receiver_cb.push_back(1);
    }
    packet_cb.push_back(1);

    // clean up semaphore in case it is reused
    noc_semaphore_set(local_semaphore_ptr, 0);
}
