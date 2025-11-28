// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/common.hpp"

#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include <cstdint>
#include <utility>
#include "tt_metal/fabric/hw/inc/linear/api.h"

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
    DPRINT << "before reserving compute cbs\n";
    cb_reserve_back(cb_id_in_l, input_num_tiles);
    DPRINT << "after reserving compute cb l\n";
    uint32_t l1_write_addr = get_write_ptr(cb_id_in_l);
    DPRINT << "compute cb l write ptr: " << (uint32_t)l1_write_addr << "\n";
    uint64_t read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_l);
    DPRINT << "read addr l: " << (uint64_t)read_addr << "\n";
    noc_async_read(read_addr, l1_write_addr, input_num_tiles * page_bytes);
    DPRINT << "after noc read l\n";
    noc_async_read_barrier();
    DPRINT << "printing local l from compute cb l\n";
    print_full_tile(cb_id_in_l, 3, false);
    cb_push_back(cb_id_in_l, input_num_tiles);

    // for tensor s
    cb_reserve_back(cb_id_in_s, onetile);
    l1_write_addr = get_write_ptr(cb_id_in_s);
    read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_s);
    noc_async_read(read_addr, l1_write_addr, onetile * page_bytes);
    noc_async_read_barrier();
    DPRINT << "printing local S from compute cb l\n";
    print_full_tile(cb_id_in_s, 0, false);
    cb_push_back(cb_id_in_s, onetile);

    // for tensor m
    cb_reserve_back(cb_id_in_m, onetile);
    l1_write_addr = get_write_ptr(cb_id_in_m);
    read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_m);
    noc_async_read(read_addr, l1_write_addr, onetile * page_bytes);
    noc_async_read_barrier();
    DPRINT << "printing local M from compute cb l\n";
    print_full_tile(cb_id_in_m, 0, false);
    cb_push_back(cb_id_in_m, onetile);
    DPRINT << "completed reading from local\n";
}

inline void read_from_int(
    uint32_t cb_int_l,
    uint32_t cb_int_s,
    uint32_t cb_int_m,
    uint32_t compute_cb_l,
    uint32_t compute_cb_s,
    uint32_t compute_cb_m,
    uint32_t onetile,
    uint32_t input_num_tiles,
    uint32_t page_bytes) {
    // mmove from intermediate cbs to compute cbs
    DPRINT << "moving from intermediate cbs to compute cbs\n";
    // for tensor l
    cb_wait_front(cb_int_l, input_num_tiles);
    DPRINT << "waiting front for l tensor\n";
    uint32_t l1_read_addr = get_read_ptr(cb_int_l);
    cb_reserve_back(compute_cb_l, input_num_tiles);
    uint32_t l1_write_addr = get_write_ptr(compute_cb_l);
    tt_memmove<false, false, false, 0>(l1_write_addr, l1_read_addr, input_num_tiles * page_bytes);
    DPRINT << "printing moved l from compute cb l\n";
    print_full_tile(compute_cb_l, 1, false);
    cb_push_back(compute_cb_l, input_num_tiles);
    cb_pop_front(cb_int_l, input_num_tiles);

    // for tensor s
    cb_wait_front(cb_int_s, onetile);
    DPRINT << "waiting front for s tensor\n";
    l1_read_addr = get_read_ptr(cb_int_s);
    cb_reserve_back(compute_cb_s, onetile);
    l1_write_addr = get_write_ptr(compute_cb_s);
    tt_memmove<false, false, false, 0>(l1_write_addr, l1_read_addr, onetile * page_bytes);
    DPRINT << "printing moved s from compute cb s\n";
    print_full_tile(compute_cb_s, 1, false);
    cb_push_back(compute_cb_s, onetile);
    cb_pop_front(cb_int_s, onetile);

    // for tensor m
    cb_wait_front(cb_int_m, onetile);
    DPRINT << "waiting front for m tensor\n";
    l1_read_addr = get_read_ptr(cb_int_m);
    cb_reserve_back(compute_cb_m, onetile);
    l1_write_addr = get_write_ptr(compute_cb_m);
    tt_memmove<false, false, false, 0>(l1_write_addr, l1_read_addr, onetile * page_bytes);
    DPRINT << "printing moved m from compute cb m\n";
    print_full_tile(compute_cb_m, 1, false);
    cb_push_back(compute_cb_m, onetile);
    cb_pop_front(cb_int_m, onetile);
}
void kernel_main() {
    DPRINT << "root reader kernel started\n";
    constexpr uint32_t fabric_ct_idx = get_compile_time_arg_val(0);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t receiver_cb_id_l = get_compile_time_arg_val(3);
    constexpr uint32_t receiver_cb_id_s = get_compile_time_arg_val(4);
    constexpr uint32_t receiver_cb_id_m = get_compile_time_arg_val(5);
    constexpr uint32_t alignment = get_compile_time_arg_val(6);
    constexpr uint32_t compute_cb_l = get_compile_time_arg_val(7);
    constexpr uint32_t compute_cb_s = get_compile_time_arg_val(8);
    constexpr uint32_t compute_cb_m = get_compile_time_arg_val(9);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(fabric_ct_idx);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(fabric_ct_idx + 1);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(fabric_ct_idx + 2);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(fabric_ct_idx + 3);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(fabric_ct_idx + 4);

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
    const auto packet_size_bytes = get_arg_val<uint32_t>(11);
    const auto page_size_bytes = get_arg_val<uint32_t>(12);
    const auto page_segments = get_arg_val<uint32_t>(13);  // always 1 delete unecessay parts
    const uint32_t sender_semaphore_addr = get_arg_val<uint32_t>(14);
    const uint32_t sender_semaphore_addr2 = get_arg_val<uint32_t>(15);
    const uint8_t sender_num_hops = get_arg_val<uint32_t>(16);  // always 1
    const uint32_t core_noc_x = get_arg_val<uint32_t>(17);
    const uint32_t core_noc_y = get_arg_val<uint32_t>(18);
    const uint32_t out_ready_sem_x = get_arg_val<uint32_t>(19);
    const uint32_t out_ready_sem_y = get_arg_val<uint32_t>(20);
    const uint32_t out_ready_sem_2_x = get_arg_val<uint32_t>(21);
    const uint32_t out_ready_sem_2_y = get_arg_val<uint32_t>(22);

    // reusing the last arg for fabric setup, therefore index overlaps.
    size_t arg_idx = 23;
    uint32_t num_tiles_l = page_idx_end;

    uint32_t chunk_size = 4;  // to be modified with tiny tiles HERE

    const uint32_t new_packet_size_bytes = packet_size_bytes + 2 * align(page_size_bytes, alignment);

    bool is_forward = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool is_termination_master = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_channel_id = get_arg_val<uint32_t>(arg_idx++);

    uint32_t termination_sync_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);

    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* mux_connection_handle;
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> mux_connection;
    mux_connection = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
        fabric_mux_x,
        fabric_mux_y,
        fabric_mux_channel_id,
        fabric_mux_num_buffers_per_channel,
        fabric_mux_channel_buffer_size_bytes,
        fabric_mux_channel_base_address,
        fabric_mux_connection_info_address,
        fabric_mux_connection_handshake_address,
        fabric_mux_flow_control_address,
        fabric_mux_buffer_index_address,
        local_flow_control_address,
        local_teardown_address,
        local_buffer_index_address);
    mux_connection_handle = &mux_connection;
    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);

    tt::tt_fabric::fabric_client_connect(*mux_connection_handle);

    cb_reserve_back(packet_header_cb_id, 1);
    const uint32_t sem_header_addr = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    const uint64_t sender_sem_noc_addr = get_noc_addr(core_noc_x, core_noc_y, sender_semaphore_addr);

    auto* sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)sem_header_ptr, sender_num_hops);

    sem_header_ptr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_sem_noc_addr, 1});

    mux_connection.wait_for_empty_write_slot();
    mux_connection.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr, packet_header_size_bytes);

    cb_reserve_back(packet_cb_id, 1);
    uint32_t packet_l1_addr = get_write_ptr(packet_cb_id);

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

    //  receive l, s and m data from sender
    auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
    noc_semaphore_wait(local_semaphore_ptr, 1);

    tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle);

    if (is_termination_master) {
        auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address);
        noc_semaphore_wait(termination_sync_ptr, num_mux_clients - 1);
        tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
    } else {
        uint64_t dest_addr =
            safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
        noc_semaphore_inc(dest_addr, 1);
        noc_async_atomic_barrier();
    }

    const uint32_t aligned_page_size_bytes = align(page_size_bytes, alignment);
    uint32_t curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx_start);
    uint32_t packet_idx = 0;  // page_idx_start / max_pages_per_packet;

    cb_reserve_back(receiver_cb_id_l, chunk_size);
    uint32_t dest_page_base_addr = get_write_ptr(receiver_cb_id_l);

    // read the single packet
    // uint64_t packet_noc_addr = packet_buffer.get_noc_addr(packet_idx, 0, 0);
    uint64_t packet_noc_addr = get_noc_addr(core_noc_x, core_noc_y, intermediate_base_addr);
    noc_async_read(packet_noc_addr, packet_l1_addr, new_packet_size_bytes);
    noc_async_read_barrier();

    tt_memmove<false, false, false, 0>(dest_page_base_addr, packet_l1_addr, packet_size_bytes);
    cb_push_back(receiver_cb_id_l, chunk_size);

    // now receiving s and m
    cb_reserve_back(receiver_cb_id_s, 1);
    cb_reserve_back(receiver_cb_id_m, 1);

    uint32_t dest_page_base_addr_s = get_write_ptr(receiver_cb_id_s);
    tt_memmove<false, false, false, 0>(
        dest_page_base_addr_s, packet_l1_addr + packet_size_bytes, aligned_page_size_bytes);
    cb_push_back(receiver_cb_id_s, 1);

    uint32_t dest_page_base_addr_m = get_write_ptr(receiver_cb_id_m);
    tt_memmove<false, false, false, 0>(
        dest_page_base_addr_m, packet_l1_addr + packet_size_bytes + aligned_page_size_bytes, aligned_page_size_bytes);
    cb_push_back(receiver_cb_id_m, 1);

    cb_push_back(packet_cb_id, 1);

    // noc_semaphore_set(local_semaphore_ptr, 0);

    // now the similar behaviour when device 2 is sending data to device 1
    // will be waiting on another semaphore, and fabric is for the other direction
    size_t fabric_idx_2_ref = fabric_idx_2;
    DPRINT << "fabric 2 rt start idx: " << (uint32_t)fabric_idx_2_ref << "\n";

    bool is_forward2 = get_arg_val<uint32_t>(fabric_idx_2_ref++) == 1;
    const bool is_termination_master2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const uint8_t fabric_mux_x2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const uint8_t fabric_mux_y2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const size_t fabric_mux_channel_base_address2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const size_t fabric_mux_connection_info_address2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const size_t fabric_mux_connection_handshake_address2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const size_t fabric_mux_flow_control_address2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const size_t fabric_mux_buffer_index_address2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    const uint8_t fabric_mux_channel_id2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);

    uint32_t termination_sync_address2 = get_semaphore(get_arg_val<uint32_t>(fabric_idx_2_ref++));
    uint32_t local_fabric_mux_status_address2 = get_semaphore(get_arg_val<uint32_t>(fabric_idx_2_ref++));
    uint32_t local_flow_control_address2 = get_semaphore(get_arg_val<uint32_t>(fabric_idx_2_ref++));
    uint32_t local_teardown_address2 = get_semaphore(get_arg_val<uint32_t>(fabric_idx_2_ref++));
    uint32_t local_buffer_index_address2 = get_semaphore(get_arg_val<uint32_t>(fabric_idx_2_ref++));

    uint32_t termination_master_noc_x2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);
    uint32_t termination_master_noc_y2 = get_arg_val<uint32_t>(fabric_idx_2_ref++);

    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* mux_connection_handle2;
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> mux_connection2;

    mux_connection2 = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
        fabric_mux_x2,
        fabric_mux_y2,
        fabric_mux_channel_id2,
        fabric_mux_num_buffers_per_channel,
        fabric_mux_channel_buffer_size_bytes,
        fabric_mux_channel_base_address2,
        fabric_mux_connection_info_address2,
        fabric_mux_connection_handshake_address2,
        fabric_mux_flow_control_address2,
        fabric_mux_buffer_index_address2,
        local_flow_control_address2,
        local_teardown_address2,
        local_buffer_index_address2);
    mux_connection_handle2 = &mux_connection2;
    DPRINT << "before waiting for fabric endpoint ready2\n";

    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x2, fabric_mux_y2, fabric_mux_status_address, local_fabric_mux_status_address);
    DPRINT << "after waiting for fabric endpoint ready2\n";
    tt::tt_fabric::fabric_client_connect(*mux_connection_handle2);

    DPRINT << "after fabric client connect2\n";
    cb_reserve_back(packet_header_cb_id, 1);
    const uint32_t sem_header_addr_2 = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    // DPRINT << "before sending semaphore inc to device 2\n";
    const uint64_t sender_sem_noc_addr_2 = get_noc_addr(core_noc_x, core_noc_y, sender_semaphore_addr2);
    auto* sem_header_ptr_2 = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr_2);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)sem_header_ptr_2, sender_num_hops);
    sem_header_ptr_2->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_sem_noc_addr_2, 1});

    mux_connection2.wait_for_empty_write_slot();
    mux_connection2.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr_2, packet_header_size_bytes);

    DPRINT << "after sending semaphore increment to device 2\n";

    // read local data from own device from intermediate buffer and push to compute cbs
    DPRINT << "indices of compute cbs: " << (uint32_t)compute_cb_l << ", " << (uint32_t)compute_cb_s << ", "
           << (uint32_t)compute_cb_m << "\n";
    DPRINT << "indices of intermediate cbs: " << (uint32_t)int_src_l << ", " << (uint32_t)int_src_s << ", "
           << (uint32_t)int_src_m << "\n";
    read_from_int(
        int_src_l, int_src_s, int_src_m, compute_cb_l, compute_cb_s, compute_cb_m, 1, chunk_size, page_size_bytes);

    DPRINT << "after reading from local second time\n";

    // read again l, s and m from device 2

    local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr2);
    noc_semaphore_wait(local_semaphore_ptr, 1);
    DPRINT << "after waiting on semaphore from device 2\n";
    tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle2);

    DPRINT << "after fabric client disconnect2\n";
    if (is_termination_master) {
        auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address2);
        noc_semaphore_wait(termination_sync_ptr, num_mux_clients - 1);
        tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x2, fabric_mux_y2, fabric_mux_termination_signal_address);
    } else {
        uint64_t dest_addr =
            safe_get_noc_addr(termination_master_noc_x2, termination_master_noc_y2, termination_sync_address2, 0);
        noc_semaphore_inc(dest_addr, 1);
        noc_async_atomic_barrier();
    }
    DPRINT << "after termination sync2\n";

    cb_reserve_back(packet_cb_id, 1);
    DPRINT << "after reserving back packet cb id\n";
    packet_l1_addr = get_write_ptr(packet_cb_id);

    page_idx_start = 0;
    curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx_start);
    packet_idx = page_idx_start / max_pages_per_packet;

    DPRINT << "RESErving back receiver cb l for chunk size: " << (uint32_t)chunk_size << "\n";
    cb_reserve_back(receiver_cb_id_l, chunk_size);
    dest_page_base_addr = get_write_ptr(receiver_cb_id_l);
    // packet_noc_addr = packet_buffer.get_noc_addr(packet_idx, 0, 0);
    packet_noc_addr = get_noc_addr(core_noc_x, core_noc_y, intermediate_base_addr);
    noc_async_read(packet_noc_addr, packet_l1_addr, new_packet_size_bytes);
    noc_async_read_barrier();
    DPRINT << "after reading packet from device 2\n";

    tt_memmove<false, false, false, 0>(dest_page_base_addr, packet_l1_addr, packet_size_bytes);
    cb_push_back(receiver_cb_id_l, chunk_size);

    cb_reserve_back(receiver_cb_id_s, 1);
    cb_reserve_back(receiver_cb_id_m, 1);
    DPRINT << "after reserving back receiver cbs s and m\n";
    dest_page_base_addr_s = get_write_ptr(receiver_cb_id_s);
    dest_page_base_addr_m = get_write_ptr(receiver_cb_id_m);

    tt_memmove<false, false, false, 0>(
        dest_page_base_addr_s, packet_l1_addr + packet_size_bytes, aligned_page_size_bytes);
    tt_memmove<false, false, false, 0>(
        dest_page_base_addr_m, packet_l1_addr + packet_size_bytes + aligned_page_size_bytes, aligned_page_size_bytes);

    DPRINT << "after memmove for s and m\n";
    cb_push_back(receiver_cb_id_s, 1);
    cb_push_back(receiver_cb_id_m, 1);

    cb_push_back(packet_cb_id, 1);
    DPRINT << "root reader kernel completed\n";
}
