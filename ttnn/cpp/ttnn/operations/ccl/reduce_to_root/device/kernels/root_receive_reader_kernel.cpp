// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/common.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using tt::data_movement::common::tt_memmove;

inline void read_from_local(
    uint32_t src_addr_l,  // source address for l tensor
    uint32_t num_tiles_l,
    uint32_t src_addr_s,  // source address for s tensor
    uint32_t src_addr_m,  // source address for m tensor
    uint32_t page_bytes,
    uint32_t core_noc_x,
    uint32_t core_noc_y,
    uint32_t cb_id_in_l,  // compute cb for l
    uint32_t cb_id_in_s,  // compute cb for s
    uint32_t cb_id_in_m,  // compute cb for m
    uint32_t onetile,
    uint32_t input_num_tiles) {
    // read l, s, m data from own device and push it to compute cbs
    DPRINT << "start reading from local\n";
    uint64_t read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_l);
    for (uint32_t i = 0; i < num_tiles_l / input_num_tiles; ++i) {
        cb_reserve_back(cb_id_in_l, input_num_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in_l);
        noc_async_read(read_addr, l1_write_addr, input_num_tiles * page_bytes);
        read_addr += input_num_tiles * page_bytes;
        noc_async_read_barrier();
        cb_push_back(cb_id_in_l, input_num_tiles);
    }

    DPRINT << "finished reading l tensor\n";
    // for tensor s
    cb_reserve_back(cb_id_in_s, onetile);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in_s);
    read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_s);
    noc_async_read(read_addr, l1_write_addr, onetile * page_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_id_in_s, onetile);

    DPRINT << "finished reading s tensor\n";
    // for tensor m
    cb_reserve_back(cb_id_in_m, onetile);
    l1_write_addr = get_write_ptr(cb_id_in_m);
    read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_m);
    noc_async_read(read_addr, l1_write_addr, onetile * page_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_id_in_m, onetile);
    DPRINT << "finished reading m tensor\n";
}

void kernel_main() {
    DPRINT << "root reader kernel started\n";
    constexpr uint32_t accessor_2_idx = get_compile_time_arg_val(0);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t receiver_cb_id_l = get_compile_time_arg_val(3);
    constexpr uint32_t receiver_cb_id_s = get_compile_time_arg_val(4);
    constexpr uint32_t receiver_cb_id_m = get_compile_time_arg_val(5);
    constexpr uint32_t alignment = get_compile_time_arg_val(6);
    constexpr uint32_t compute_cb_l = get_compile_time_arg_val(7);
    constexpr uint32_t compute_cb_s = get_compile_time_arg_val(8);
    constexpr uint32_t compute_cb_m = get_compile_time_arg_val(9);
    constexpr uint32_t core_noc_x = get_compile_time_arg_val(10);
    constexpr uint32_t core_noc_y = get_compile_time_arg_val(11);
    constexpr auto packet_buffer_args = TensorAccessorArgs<12>();
    constexpr auto packet_buffer_args_2 = TensorAccessorArgs<accessor_2_idx>();

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    const uint32_t fabric_idx_2 = get_arg_val<uint32_t>(0);
    const uint32_t src_addr_l = get_arg_val<uint32_t>(1);
    const uint32_t src_addr_s = get_arg_val<uint32_t>(2);
    const uint32_t src_addr_m = get_arg_val<uint32_t>(3);
    const uint32_t int_src_l = get_arg_val<uint32_t>(4);
    const uint32_t int_src_s = get_arg_val<uint32_t>(5);
    const uint32_t int_src_m = get_arg_val<uint32_t>(6);
    auto page_idx_start = get_arg_val<uint32_t>(7);
    const auto page_idx_end = get_arg_val<uint32_t>(8);
    const auto max_pages_per_packet = get_arg_val<uint32_t>(9);
    const auto intermediate_base_addr = get_arg_val<uint32_t>(10);
    const auto intermediate_base_addr_2 = get_arg_val<uint32_t>(11);
    const auto packet_size_bytes = get_arg_val<uint32_t>(12);
    const auto page_size_bytes = get_arg_val<uint32_t>(13);
    const auto page_segments = get_arg_val<uint32_t>(14);
    const uint32_t sender_semaphore_addr = get_arg_val<uint32_t>(15);
    const uint32_t sender_semaphore_addr2 = get_arg_val<uint32_t>(16);
    const uint8_t sender_num_hops = get_arg_val<uint32_t>(17);  // always 1
    const bool sender_is_forward = get_arg_val<uint32_t>(18);

    // reusing the last arg for fabric setup, therefore index overlaps.
    size_t conn_arg_idx = 18;
    uint32_t num_tiles_l = page_idx_end;

    uint32_t chunk_size = 2;  // to be modified with tiny tiles HERE

    const auto packet_buffer = TensorAccessor(packet_buffer_args, intermediate_base_addr, packet_size_bytes);
    const auto packet_buffer_2 = TensorAccessor(packet_buffer_args_2, intermediate_base_addr_2, packet_size_bytes);

    auto fabric_connection = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(conn_arg_idx);

    const bool sender_is_forward2 = get_arg_val<uint32_t>(fabric_idx_2);

    cb_reserve_back(packet_header_cb_id, 1);
    const uint32_t sem_header_addr = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

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

    cb_reserve_back(packet_cb_id, 1);
    const uint64_t packet_l1_addr = get_write_ptr(packet_cb_id);

    // read local data from own device and push to compute cbs
    read_from_local(
        src_addr_l,
        num_tiles_l,
        src_addr_s,
        src_addr_m,
        page_size_bytes,
        core_noc_x,
        core_noc_y,
        compute_cb_l,
        compute_cb_s,
        compute_cb_m,
        1,
        chunk_size);
    // device 0 is sending data to device 1

    DPRINT << "after reading from local\n";
    // receive l, s and m data from sender
    auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
    noc_semaphore_wait(local_semaphore_ptr, 1);

    const uint32_t aligned_page_size_bytes = align(page_size_bytes, alignment);
    uint32_t curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx_start);
    uint32_t packet_idx = page_idx_start / max_pages_per_packet;

    cb_reserve_back(receiver_cb_id_l, chunk_size);
    uint32_t dest_page_base_addr = get_write_ptr(receiver_cb_id_l);
    uint32_t count = 0;
    for (uint32_t page_idx = page_idx_start, packet_page_idx = 0; page_idx < page_idx_end; ++page_idx) {
        for (uint32_t page_segment_idx = 0; page_segment_idx < page_segments; ++page_segment_idx) {
            if (page_idx == page_idx_start || packet_page_idx == curr_pages_per_packet) {
                const uint64_t packet_noc_addr = packet_buffer.get_noc_addr(packet_idx, 0, 0);
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
            dest_page_base_addr += aligned_page_size_bytes;
        }
        count++;
        if (count == chunk_size || page_idx == page_idx_end - 1) {
            cb_push_back(receiver_cb_id_l, count);
            count = 0;
            dest_page_base_addr = get_write_ptr(receiver_cb_id_l);
            if (page_idx != page_idx_end - 1) {
                cb_reserve_back(receiver_cb_id_l, chunk_size);
            }
        }
    }
    cb_push_back(packet_cb_id, 1);
    DPRINT << "after receiving packet l\n";

    // now receiving s and m
    cb_reserve_back(receiver_cb_id_s, 1);
    cb_reserve_back(receiver_cb_id_m, 1);
    const uint32_t dest_page_base_addr_s2 = get_write_ptr(receiver_cb_id_s);
    const uint32_t dest_page_base_addr_m2 = get_write_ptr(receiver_cb_id_m);

    auto packet_idx_sm = 0;
    const uint64_t pkt_noc_addr2 = packet_buffer_2.get_noc_addr(packet_idx_sm, 0, 0);
    // read the single packet that contains both s and m to a temporary buffer
    // then copy first tile to s and second tile to m
    noc_async_read(pkt_noc_addr2, packet_l1_addr, page_size_bytes * 2);
    noc_async_read_barrier();

    tt_memmove<false, false, false, 0>(dest_page_base_addr_s2, packet_l1_addr, page_size_bytes);
    tt_memmove<false, false, false, 0>(
        dest_page_base_addr_m2, packet_l1_addr + aligned_page_size_bytes, page_size_bytes);

    noc_semaphore_set(local_semaphore_ptr, 0);

    DPRINT << "after receiving packet s and m\n";

    // now the similar behaviour when device 2 is sending data to device 1
    // will be waiting on another semaphore, and fabric is for the other direction
    size_t fabric_idx_2_ref = fabric_idx_2;
    auto fabric_connection_2 = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(fabric_idx_2_ref);
    cb_reserve_back(packet_header_cb_id, 1);
    const uint32_t sem_header_addr_2 = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    const uint64_t sender_sem_noc_addr_2 = get_noc_addr(sender_semaphore_addr2);
    auto* sem_header_ptr_2 = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr_2);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)sem_header_ptr_2, sender_num_hops);
    sem_header_ptr_2->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_sem_noc_addr_2, 1});
    fabric_connection_2.open_finish();
    auto& connection_direction_2 =
        tt::point_to_point::common::connection_direction_collection(sender_is_forward2, fabric_connection_2);
    connection_direction_2.wait_for_empty_write_slot();
    connection_direction_2.send_payload_flush_blocking_from_address(
        (uint32_t)sem_header_ptr_2, packet_header_size_bytes);
    fabric_connection_2.close();

    DPRINT << "after sending semaphore increment to device 2\n";

    // read local data from own device from intermediate buffer and push to compute cbs
    read_from_local(
        get_read_ptr(int_src_l),
        num_tiles_l,
        get_read_ptr(int_src_s),
        get_read_ptr(int_src_m),
        page_size_bytes,
        core_noc_x,
        core_noc_y,
        compute_cb_l,
        compute_cb_s,
        compute_cb_m,
        1,
        chunk_size);

    DPRINT << "after reading from local second time\n";

    // read again l, s and m from device 2

    local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr2);
    noc_semaphore_wait(local_semaphore_ptr, 1);

    DPRINT << "after waiting on semaphore from device 2\n";

    page_idx_start = 0;
    curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx_start);
    packet_idx = page_idx_start / max_pages_per_packet;

    cb_reserve_back(receiver_cb_id_l, chunk_size);
    dest_page_base_addr = get_write_ptr(receiver_cb_id_l);
    count = 0;
    for (uint32_t page_idx = page_idx_start, packet_page_idx = 0; page_idx < page_idx_end; ++page_idx) {
        for (uint32_t page_segment_idx = 0; page_segment_idx < page_segments; ++page_segment_idx) {
            if (page_idx == page_idx_start || packet_page_idx == curr_pages_per_packet) {
                const uint64_t packet_noc_addr = packet_buffer.get_noc_addr(packet_idx, 0, 0);
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
            dest_page_base_addr += aligned_page_size_bytes;
        }
        count++;
        if (count == chunk_size || page_idx == page_idx_end - 1) {
            cb_push_back(receiver_cb_id_l, count);
            count = 0;
            dest_page_base_addr = get_write_ptr(receiver_cb_id_l);
            if (page_idx != page_idx_end - 1) {
                cb_reserve_back(receiver_cb_id_l, chunk_size);
            }
        }
    }
    cb_push_back(packet_cb_id, 1);

    // now receiving s and m
    cb_reserve_back(receiver_cb_id_s, 1);
    cb_reserve_back(receiver_cb_id_m, 1);
    const uint32_t dest_page_base_addr_s = get_write_ptr(receiver_cb_id_s);
    const uint32_t dest_page_base_addr_m = get_write_ptr(receiver_cb_id_m);

    packet_idx_sm = 0;
    const uint64_t packet_noc_addr2 = packet_buffer_2.get_noc_addr(packet_idx_sm, 0, 0);
    // read the single packet that contains both s and m to a temporary buffer
    // then copy first tile to s and second tile to m
    noc_async_read(packet_noc_addr2, packet_l1_addr, page_size_bytes * 2);
    noc_async_read_barrier();

    tt_memmove<false, false, false, 0>(dest_page_base_addr_s, packet_l1_addr, page_size_bytes);
    tt_memmove<false, false, false, 0>(
        dest_page_base_addr_m, packet_l1_addr + aligned_page_size_bytes, page_size_bytes);

    noc_semaphore_set(local_semaphore_ptr, 0);
    DPRINT << "root reader kernel completed\n";
}
