// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"

using tt::data_movement::common::round_up;
using tt::data_movement::common::tt_memmove;

// device 2 writer receives data from compute kernel and sends it to device 1
void kernel_main() {
    constexpr uint32_t accessor_2_idx = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_l = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_s = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_m = get_compile_time_arg_val(3);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t alignment = get_compile_time_arg_val(6);
    constexpr uint32_t core_noc_x = get_compile_time_arg_val(7);
    constexpr uint32_t core_noc_y = get_compile_time_arg_val(8);
    constexpr auto dst_buffer_args = TensorAccessorArgs<9>();
    constexpr auto dst_buffer_2_args = TensorAccessorArgs<accessor_2_idx>();

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    const uint32_t receiver_base_address = get_arg_val<uint32_t>(0);
    const uint32_t receiver_base_address_2 = get_arg_val<uint32_t>(1);
    const auto page_idx_start = get_arg_val<uint32_t>(2);
    const auto page_idx_end = get_arg_val<uint32_t>(3);
    const uint8_t dst_num_hops = 1;  // get_arg_val<uint32_t>(3);
    const auto page_size_bytes = get_arg_val<uint32_t>(4);
    const auto payload_size_bytes = get_arg_val<uint32_t>(5);
    // send a single packet for l tensor (8 pages)
    // send a single packet for m and s tensors (2 pages: 1 each)
    const auto max_pages_per_packet_l = 8;  // get_arg_val<uint32_t>(6);
    const auto max_pages_per_packet_ms = 2;
    const auto page_segments = get_arg_val<uint32_t>(6);
    const uint32_t receive_semaphore_addr = get_arg_val<uint32_t>(7);
    const bool dst_is_forward = get_arg_val<uint32_t>(8);

    const uint32_t aligned_page_size_bytes = round_up(page_size_bytes, alignment);

    // reusing the last arg for fabric setup, therefore index overlaps.
    size_t conn_arg_idx = 8;
    uint32_t chunk_size = 8;

    auto fabric_connection = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(conn_arg_idx);

    // set up packet header buffer
    cb_reserve_back(packet_header_cb_id, 1);
    uint32_t packet_header_addr = get_read_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    auto* packet_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)packet_header_ptr, dst_num_hops);

    const auto dst_buffer = TensorAccessor(dst_buffer_args, receiver_base_address, payload_size_bytes);
    const auto dst_buffer_2 = TensorAccessor(dst_buffer_2_args, receiver_base_address_2, payload_size_bytes);

    // working memory to hold coalesced packet
    cb_reserve_back(packet_cb_id, 1);
    const uint32_t packet_base_addr = get_write_ptr(packet_cb_id);
    cb_push_back(packet_cb_id, 1);

    // initial packet size
    uint32_t curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx_start);
    uint32_t packet_idx = page_idx_start / max_pages_per_packet;

    // wait for receiver to signal it is ready
    auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive_semaphore_addr);
    noc_semaphore_wait(local_semaphore_ptr, 1);
    // clean up semaphore – needs to be done before the sender side semaphore increment if we're re-using the semaphore
    // in subsequent program cache hits
    noc_semaphore_set(local_semaphore_ptr, 0);

    fabric_connection.open_finish();
    auto& connection_direction =
        tt::point_to_point::common::connection_direction_collection(dst_is_forward, fabric_connection);

    // uint64_t dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, receiver_base_address, 0);
    for (uint32_t page_idx = page_idx_start, packet_page_idx = 0; page_idx < page_idx_end; page_idx += chunk_size) {
        cb_wait_front(cb_id_l, chunk_size);
        const uint32_t src_page_base_addr = get_read_ptr(cb_id_l);
        for (uint32_t chunk_offset = 0; chunk_offset < std::min(chunk_size, page_idx_end - page_idx); ++chunk_offset) {
            for (uint32_t page_segment_idx = 0; page_segment_idx < page_segments; ++page_segment_idx) {
                const uint32_t page_offset = page_segment_idx * payload_size_bytes;
                const uint32_t src_addr = src_page_base_addr + page_offset;
                const uint32_t transfer_size_bytes = std::min(page_size_bytes - page_offset, payload_size_bytes);

                // copy page to packet buffer with offset
                const uint32_t packet_addr = packet_base_addr + packet_page_idx * aligned_page_size_bytes;
                tt_memmove<false, false, false, 0>(packet_addr, src_addr, transfer_size_bytes);
                ++packet_page_idx;
                if (packet_page_idx >= curr_pages_per_packet) {
                    auto packet_size_bytes = align(payload_size_bytes, alignment);
                    // packet_header_ptr->to_noc_unicast_write(NocUnicastCommandHeader{dst_noc_addr},
                    // align(payload_size_bytes, alignment)); perform_payload_send(connection_direction,
                    // packet_base_addr, payload_size_bytes, packet_header_ptr); dst_noc_addr += packet_size_bytes;

                    const uint64_t dst_noc_addr = get_noc_addr(packet_idx, dst_buffer, 0, 0);
                    tt::tt_fabric::linear::to_noc_unicast_write(
                        align(payload_size_bytes, alignment), packet_header_ptr, packet_idx, dst_buffer);
                    perform_payload_send(connection_direction, packet_base_addr, payload_size_bytes, packet_header_ptr);

                    // reset counters
                    packet_page_idx = 0;
                    curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx - 1);

                    ++packet_idx;
                }
            }
            src_page_base_addr += aligned_page_size_bytes;
        }
        cb_pop_front(cb_id_l, chunk_size);
    }

    cb_reserve_back(packet_header_cb_id, 1);
    const uint32_t sem_header_addr = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    // now send m and s in a single packet
    // read from both cb to a temporary buffer and send
    cb_wait_front(cb_id_m, 1);
    const uint32_t src_page_base_addr_m = get_read_ptr(cb_id_m);
    tt_memmove<false, false, false, 0>(packet_base_addr, src_page_base_addr_m, page_size_bytes);
    cb_pop_front(cb_id_m, 1);
    cb_wait_front(cb_id_s, 1);
    const uint32_t src_page_base_addr_s = get_read_ptr(cb_id_s);
    tt_memmove<false, false, false, 0>(
        packet_base_addr + aligned_page_size_bytes, src_page_base_addr_s, page_size_bytes);
    cb_pop_front(cb_id_s, 1);
    const uint32_t total_payload_size_ms = 2 * payload_size_bytes;
    const uint32_t packet_size_bytes_ms = align(total_payload_size_ms, alignment);
    // packet_header_ptr->to_noc_unicast_write(NocUnicastCommandHeader{dst_noc_addr}, packet_size_bytes_ms);
    // perform_payload_send(connection_direction, packet_base_addr, total_payload_size_ms, packet_header_ptr);

    uint32_t packet_idx_sm = 0;  // separate index for m and s packet
    const uint64_t dst_noc_addr2 = get_noc_addr(packet_idx_sm, dst_buffer2, 0, 0);
    tt::tt_fabric::linear::to_noc_unicast_write(
        align(payload_size_bytes, alignment), packet_header_ptr, packet_idx_sm, dst_buffer_2);
    perform_payload_send(connection_direction, packet_base_addr, payload_size_bytes, packet_header_ptr);

    const uint64_t receive_sem_noc_addr = get_noc_addr(receive_semaphore_addr);

    auto* sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)sem_header_ptr, dst_num_hops);
    sem_header_ptr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{receive_sem_noc_addr, 1});

    connection_direction.wait_for_empty_write_slot();
    connection_direction.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr, packet_header_size_bytes);

    fabric_connection.close();
}
