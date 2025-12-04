// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/common.hpp"

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
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
    // DPRINT << "pushing first set of inputs for compute to cbs " << (uint32_t)cb_id_in_l << ", " <<
    // (uint32_t)cb_id_in_s
    //        << ", " << (uint32_t)cb_id_in_m << "\n";
    //  for tensor l
    cb_reserve_back(cb_id_in_l, input_num_tiles);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in_l);
    uint64_t read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_l);
    // //DPRINT << "read addr l: " << (uint64_t)read_addr << "\n";
    noc_async_read(read_addr, l1_write_addr, input_num_tiles * page_bytes);
    // noc_async_read_barrier();
    // DPRINT << "printing local l from compute cb l\n";
    // print_full_tile(cb_id_in_l, 3, false);
    cb_push_back(cb_id_in_l, input_num_tiles);

    // for tensor s
    cb_reserve_back(cb_id_in_s, onetile);
    l1_write_addr = get_write_ptr(cb_id_in_s);
    read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_s);
    // //DPRINT << "read addr s: " << (uint64_t)read_addr << "\n";
    noc_async_read(read_addr, l1_write_addr, onetile * page_bytes);
    // noc_async_read_barrier();
    // DPRINT << "printing local s from compute cb s\n";
    // print_full_tile(cb_id_in_s, 0, false);
    cb_push_back(cb_id_in_s, onetile);

    // for tensor m
    cb_reserve_back(cb_id_in_m, onetile);
    l1_write_addr = get_write_ptr(cb_id_in_m);
    read_addr = get_noc_addr(core_noc_x, core_noc_y, src_addr_m);
    // //DPRINT << "read addr m: " << (uint64_t)read_addr << "\n";
    noc_async_read(read_addr, l1_write_addr, onetile * page_bytes);
    noc_async_read_barrier();
    // DPRINT << "printing local m from compute cb m\n";
    // print_full_tile(cb_id_in_m, 0, false);
    cb_push_back(cb_id_in_m, onetile);
}

void kernel_main() {
    // DPRINT << "root2 reader kernel started\n";
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
    constexpr uint32_t input_num_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t packet_size_bytes = get_compile_time_arg_val(12);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    const uint32_t src_addr_l = get_arg_val<uint32_t>(0);
    const uint32_t src_addr_s = get_arg_val<uint32_t>(1);
    const uint32_t src_addr_m = get_arg_val<uint32_t>(2);
    const auto intermediate_base_addr = get_arg_val<uint32_t>(3);
    const uint32_t sender_semaphore_addr = get_arg_val<uint32_t>(4);
    const uint32_t core_noc_x = get_arg_val<uint32_t>(5);
    const uint32_t core_noc_y = get_arg_val<uint32_t>(6);
    const auto page_idx_start = 0;
    const auto page_idx_end = input_num_tiles;
    const auto max_pages_per_packet = input_num_tiles;
    const uint8_t sender_num_hops = 1;

    // reusing the last arg for fabric setup, therefore index overlaps.
    size_t arg_idx = 7;

    const uint32_t new_packet_size_bytes = packet_size_bytes + 2 * align(page_size_bytes, alignment);

    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(fabric_ct_idx);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(fabric_ct_idx + 1);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(fabric_ct_idx + 2);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(fabric_ct_idx + 3);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(fabric_ct_idx + 4);

    // //DPRINT << "after packet buffer setup\n";

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
    // //DPRINT << "fabric mux x: " << (uint32_t)fabric_mux_x << ", y: " << (uint32_t)fabric_mux_y
    //        << ", channel id: " << (uint32_t)fabric_mux_channel_id << "\n";

    // //DPRINT << "is termination master: " << (uint32_t)is_termination_master << "\n";

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
    // //DPRINT << "after fabric client connect\n";

    cb_reserve_back(packet_header_cb_id, 1);
    const uint32_t sem_header_addr = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    // //DPRINT << "before sending semaphore inc\n";

    const uint64_t sender_sem_noc_addr = get_noc_addr(core_noc_x, core_noc_y, sender_semaphore_addr);
    // //DPRINT << "SEMAPHORE ADDRESS IS: " << (uint64_t)sender_sem_noc_addr << "with core noc x:" <<
    // (uint32_t)core_noc_x
    //        << "and core noc y: " << (uint32_t)core_noc_y << "\n";
    //  const uint64_t sender_sem_noc_addr = safe_get_noc_addr(out_ready_sem_x, out_ready_sem_y, sender_semaphore_addr,
    //  0);

    auto* sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)sem_header_ptr, sender_num_hops);
    sem_header_ptr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sender_sem_noc_addr, 1});

    mux_connection.wait_for_empty_write_slot();
    mux_connection.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr, packet_header_size_bytes);
    // //DPRINT << "after sending semaphore incremement\n";

    // //DPRINT << "before reserving back packet cb\n";
    // //DPRINT << "the packet cb id: " << (uint32_t)packet_cb_id << "\n";
    cb_reserve_back(packet_cb_id, 1);
    // //DPRINT << "reserving back packet cb\n";
    const uint32_t packet_l1_addr = get_write_ptr(packet_cb_id);

    // //DPRINT << "before reading from local\n";
    //  read local data from own device and push to compute cbs
    read_from_local(
        src_addr_l,
        page_idx_end,
        src_addr_s,
        src_addr_m,
        page_size_bytes,
        core_noc_x,
        core_noc_y,
        compute_cb_l,
        compute_cb_s,
        compute_cb_m,
        1,
        input_num_tiles);

    // device 3 is sending data to device 2
    uint32_t chunk_size = input_num_tiles;  // 8; HERE

    // here fix semaphore here as well
    //  receive l, s and m data from sender
    auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore_addr);
    noc_semaphore_wait(local_semaphore_ptr, 1);
    // //DPRINT << "after waiting on semaphore\n";
    noc_semaphore_set(local_semaphore_ptr, 0);

    tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle);
    // //DPRINT << "after fabric client disconnect\n";
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
    // //DPRINT << "after termination sync\n";

    const uint32_t aligned_page_size_bytes = align(page_size_bytes, alignment);
    // DPRINT << "aligned page size bytes: " << (uint32_t)aligned_page_size_bytes << "\n";
    // DPRINT << "PAGE SIZE BYTES: " << (uint32_t)page_size_bytes << "\n";
    // DPRINT << "packet size bytes: " << (uint32_t)packet_size_bytes << "\n";
    // DPRINT << "aligned packet size bytes: " << (uint32_t)align(packet_size_bytes, alignment) << "\n";
    uint32_t curr_pages_per_packet = std::min(max_pages_per_packet, page_idx_end - page_idx_start);
    uint32_t packet_idx = page_idx_start / max_pages_per_packet;

    cb_reserve_back(receiver_cb_id_l, chunk_size);
    uint32_t dest_page_base_addr = get_write_ptr(receiver_cb_id_l);

    // const uint64_t packet_noc_addr = packet_buffer.get_noc_addr(packet_idx, 0, 0);
    const uint64_t packet_noc_addr = get_noc_addr(core_noc_x, core_noc_y, intermediate_base_addr);
    // //DPRINT << "reading packet from address: " << (uint32_t)packet_noc_addr << "\n";
    // //DPRINT << "reading size: " << (uint32_t)new_packet_size_bytes << "\n";
    noc_async_read(packet_noc_addr, packet_l1_addr, new_packet_size_bytes);
    noc_async_read_barrier();

    tt_memmove<false, false, false, 0>(dest_page_base_addr, packet_l1_addr, packet_size_bytes);
    // DPRINT << "print received l from receiver cb l\n";
    // print_full_tile(receiver_cb_id_l, 2, false);
    cb_push_back(receiver_cb_id_l, chunk_size);

    // now receiving s and m
    // DPRINT << "pushing second set of inputs for compute to cbs " << (uint32_t)receiver_cb_id_l << ", "
    //       << (uint32_t)receiver_cb_id_s << ", " << (uint32_t)receiver_cb_id_m << "\n";
    cb_reserve_back(receiver_cb_id_s, 1);
    cb_reserve_back(receiver_cb_id_m, 1);

    const uint32_t dest_page_base_addr_s = get_write_ptr(receiver_cb_id_s);
    const uint32_t dest_page_base_addr_m = get_write_ptr(receiver_cb_id_m);

    tt_memmove<false, false, false, 0>(
        dest_page_base_addr_s, packet_l1_addr + packet_size_bytes, aligned_page_size_bytes);
    tt_memmove<false, false, false, 0>(
        dest_page_base_addr_m, packet_l1_addr + packet_size_bytes + aligned_page_size_bytes, aligned_page_size_bytes);

    // DPRINT << "print received s from receiver cb s\n";
    // print_full_tile(receiver_cb_id_s, 0, false);
    // DPRINT << "print received m from receiver cb m\n";
    // print_full_tile(receiver_cb_id_m, 0, false);

    cb_push_back(receiver_cb_id_s, 1);
    cb_push_back(receiver_cb_id_m, 1);

    // //DPRINT << "printing the packet data we have received first time\n";
    cb_push_back(packet_cb_id, 1);

    // //DPRINT << "after memmove of s and m\n";
    // DPRINT << "ROOT 2 reader kernel completed\n";
}
