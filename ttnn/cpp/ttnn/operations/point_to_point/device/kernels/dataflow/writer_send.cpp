
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

#include "dprint_pages.h"

using tt::data_movement::common::round_up;
using tt::data_movement::common::tt_memmove;

inline auto& connection_direction_collection(const bool dst_is_forward, FabricConnectionManager& fabric_connection) {
    if (dst_is_forward) {
        return fabric_connection.get_forward_connection();
    } else {
        return fabric_connection.get_backward_connection();
    }
}

void kernel_main() {
    constexpr uint32_t sender_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(2);
    constexpr bool dst_is_dram = get_compile_time_arg_val(3);
    constexpr uint32_t alignment = get_compile_time_arg_val(4);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
    constexpr uint32_t noc_idx = 0;

    const uint32_t receiver_base_address = get_arg_val<uint32_t>(0);
    const auto page_idx_start = get_arg_val<uint32_t>(1);
    const auto page_idx_end = get_arg_val<uint32_t>(2);
    const uint8_t dst_num_hops = get_arg_val<uint32_t>(3);
    const auto page_size_bytes = get_arg_val<uint32_t>(4);
    const auto payload_size_bytes = get_arg_val<uint32_t>(5);
    const auto max_pages_per_packet = get_arg_val<uint32_t>(6);
    const uint32_t receive_semaphore_addr = get_arg_val<uint32_t>(7);
    const bool dst_is_forward = get_arg_val<uint32_t>(8);

    const uint32_t aligned_page_size_bytes = round_up(page_size_bytes, alignment);

    // reusing the last arg for fabric setup, therefore index overlaps.
    size_t conn_arg_idx = 8;
    auto fabric_connection = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(conn_arg_idx);

    // set up packet header buffer
    cb_reserve_back(packet_header_cb_id, 1);
    uint32_t packet_header_addr = get_read_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    auto* packet_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
    packet_header_ptr->to_chip_unicast(dst_num_hops);

    InterleavedAddrGen<dst_is_dram> dst_buffer_addrgen{
        .bank_base_address = receiver_base_address, .page_size = payload_size_bytes};

    // working memory to hold coalesced packet
    cb_reserve_back(packet_cb_id, 1);
    const uint32_t packet_base_addr = get_write_ptr(packet_cb_id);
    cb_push_back(packet_cb_id, 1);

    // initial packet size
    uint32_t curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx_start);
    uint32_t packet_idx = page_idx_start / max_pages_per_packet;

    fabric_connection.open_finish();
    auto& connection_direction = connection_direction_collection(dst_is_forward, fabric_connection);

    for (uint32_t page_idx = page_idx_start, packet_page_idx = 0; page_idx < page_idx_end;
         ++page_idx, ++packet_page_idx) {
        cb_wait_front(sender_cb_id, 1);
        const uint32_t src_page_ptr = get_read_ptr(sender_cb_id);

        // copy page to packet buffer with offset
        const uint32_t packet_addr = packet_base_addr + packet_page_idx * aligned_page_size_bytes;
        tt_memmove<false, false, false, 0>(packet_addr, src_page_ptr, page_size_bytes);

        // !TODO better use of async copies
        cb_pop_front(sender_cb_id, 1);

        if (packet_page_idx == curr_pages_per_packet - 1) {
            const uint64_t dst_noc_addr = get_noc_addr(packet_idx, dst_buffer_addrgen, 0 /*offset*/, 0 /*noc_id*/);

            packet_header_ptr->to_noc_unicast_write(
                tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr}, payload_size_bytes);

            connection_direction.wait_for_empty_write_slot();
            connection_direction.send_payload_without_header_non_blocking_from_address(
                packet_base_addr, payload_size_bytes);
            connection_direction.send_payload_blocking_from_address(
                (uint32_t)packet_header_ptr, packet_header_size_bytes);

            // reset counters
            packet_page_idx = 0;
            ++packet_idx;
            curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx);
        }
    }

    cb_reserve_back(packet_header_cb_id, 1);
    const uint32_t sem_header_addr = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    const uint64_t receive_sem_noc_addr = get_noc_addr(receive_semaphore_addr);

    auto* sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr);
    sem_header_ptr->to_chip_unicast(dst_num_hops);
    sem_header_ptr->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{receive_sem_noc_addr, 1, 32});

    connection_direction.wait_for_empty_write_slot();
    connection_direction.send_payload_blocking_from_address((uint32_t)sem_header_ptr, packet_header_size_bytes);

    fabric_connection.close();
}
