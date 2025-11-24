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

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint8_t r = 0; r < 8; ++r) {
        SliceRange sr_left = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 16, .ws = 1};
        SliceRange sr_right =
            SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 17, .w1 = 32, .ws = 1};
        DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
               << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

void kernel_main() {
    DPRINT << "sender writer kernel started\n";
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

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    const uint32_t receiver_base_address = get_arg_val<uint32_t>(0);
    const uint32_t page_idx_start = get_arg_val<uint32_t>(1);
    const uint32_t page_idx_end = get_arg_val<uint32_t>(2);
    const uint8_t dst_num_hops = 1;  // get_arg_val<uint32_t>(3);
    const auto page_size_bytes = get_arg_val<uint32_t>(3);
    const auto payload_size_bytes = get_arg_val<uint32_t>(4);
    // send a single packet for l tensor (8 pages)
    // send a single packet for m and s tensors (2 pages: 1 each)
    const uint32_t max_pages_per_packet_l = 4;  // 2;  // get_arg_val<uint32_t>(6); //HERE
    const uint32_t max_pages_per_packet_ms = 2;
    const auto page_segments = get_arg_val<uint32_t>(5);  // always 1 delete unecessay parts
    const uint32_t receive_semaphore_addr = get_arg_val<uint32_t>(6);
    const bool dst_is_forward = get_arg_val<uint32_t>(7);

    const uint32_t aligned_page_size_bytes = round_up(page_size_bytes, alignment);

    const uint32_t new_payload_size_bytes =
        payload_size_bytes + 2 * aligned_page_size_bytes;  // add the extra size for s and m
    DPRINT << "new payload size bytes: " << (uint32_t)new_payload_size_bytes << ENDL();

    // reusing the last arg for fabric setup, therefore index overlaps.
    size_t conn_arg_idx = 7;

    auto fabric_connection = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(conn_arg_idx);

    // set up packet header buffer
    cb_reserve_back(packet_header_cb_id, 1);
    uint32_t packet_header_addr = get_read_ptr(packet_header_cb_id);
    // cb_push_back(packet_header_cb_id, 1);

    auto* packet_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)packet_header_ptr, dst_num_hops);

    const auto dst_buffer = TensorAccessor(dst_buffer_args, receiver_base_address, new_payload_size_bytes);

    // working memory to hold coalesced packet
    cb_reserve_back(packet_cb_id, 1);
    uint32_t packet_base_addr = get_write_ptr(packet_cb_id);
    // cb_push_back(packet_cb_id, 1);

    // initial packet size
    uint32_t curr_pages_per_packet = std::min(max_pages_per_packet_l, page_idx_end - page_idx_start);
    uint32_t packet_idx = page_idx_start / max_pages_per_packet_l;

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

    for (uint32_t page_idx = page_idx_start, packet_page_idx = 0; page_idx < page_idx_end; ++page_idx) {
        cb_wait_front(cb_id_l, 1);
        uint32_t src_page_base_addr = get_read_ptr(cb_id_l);
        print_full_tile(cb_id_l, 0, false);
        const uint32_t src_addr = src_page_base_addr;
        const uint32_t transfer_size_bytes = std::min(page_size_bytes, payload_size_bytes);

        // copy page to packet buffer with offset
        const uint32_t packet_addr = packet_base_addr + packet_page_idx * aligned_page_size_bytes;
        tt_memmove<false, false, false, 0>(packet_addr, src_addr, transfer_size_bytes);
        ++packet_page_idx;
        if (packet_page_idx >= curr_pages_per_packet) {
            // add s and m data to the packet before sending it
            DPRINT << "adding s and m data to the packet before sending it\n";
            cb_wait_front(cb_id_m, 1);
            const uint32_t src_page_base_addr_m = get_read_ptr(cb_id_m);
            uint32_t packet_m_addr = packet_base_addr + packet_page_idx * aligned_page_size_bytes;
            tt_memmove<false, false, false, 0>(packet_m_addr, src_page_base_addr_m, page_size_bytes);
            cb_pop_front(cb_id_m, 1);
            cb_wait_front(cb_id_s, 1);
            const uint32_t src_page_base_addr_s = get_read_ptr(cb_id_s);
            tt_memmove<false, false, false, 0>(
                packet_m_addr + aligned_page_size_bytes, src_page_base_addr_s, page_size_bytes);
            cb_pop_front(cb_id_s, 1);
            DPRINT << "finished adding s and m data to the packet\n";
            const uint64_t dst_noc_addr = dst_buffer.get_noc_addr(packet_idx, 0, 0);
            DPRINT << "before NOC UNICAST WRITE\n";
            tt::tt_fabric::linear::to_noc_unicast_write(
                align(new_payload_size_bytes, alignment), packet_header_ptr, packet_idx, dst_buffer);
            DPRINT << "AFTER NOC UNICAST WRITE\n";
            perform_payload_send(connection_direction, packet_base_addr, new_payload_size_bytes, packet_header_ptr);

            DPRINT << "performing packet send for packet idx: " << (uint32_t)packet_idx << ENDL();
            // reset counters
            packet_page_idx = 0;
            curr_pages_per_packet = std::min(max_pages_per_packet_l, page_idx_end - page_idx - 1);

            ++packet_idx;
        }
        cb_pop_front(cb_id_l, 1);
    }

    DPRINT << "after loop\n";

    cb_reserve_back(packet_header_cb_id, 1);
    uint32_t sem_header_addr = get_read_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    const uint64_t receive_sem_noc_addr = get_noc_addr(receive_semaphore_addr);

    auto* sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)sem_header_ptr, dst_num_hops);
    sem_header_ptr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{receive_sem_noc_addr, 1});

    connection_direction.wait_for_empty_write_slot();
    connection_direction.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr, packet_header_size_bytes);

    fabric_connection.close();
    DPRINT << "sender writer kernel completed\n";
}
