// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include <cstdint>
#include <utility>
#include "tt_metal/fabric/hw/inc/linear/api.h"

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
// device 2 writer receives data from compute kernel and sends it to device 1
void kernel_main() {
    DPRINT << "root2 writer kernel started\n";
    constexpr uint32_t fabric_ct_idx = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_l = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_s = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_m = get_compile_time_arg_val(3);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t alignment = get_compile_time_arg_val(6);
    constexpr uint32_t input_num_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t payload_size_bytes = get_compile_time_arg_val(9);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(fabric_ct_idx);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(fabric_ct_idx + 1);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(fabric_ct_idx + 2);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(fabric_ct_idx + 3);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(fabric_ct_idx + 4);

    DPRINT << "FABRIC MUX CT ARGS: \n";
    DPRINT << "num buffers per channel: " << (uint32_t)fabric_mux_num_buffers_per_channel << "\n";
    DPRINT << "channel buffer size bytes: " << (uint32_t)fabric_mux_channel_buffer_size_bytes << "\n";
    DPRINT << "status address: " << (uint32_t)fabric_mux_status_address << "\n";
    DPRINT << "termination signal address: " << (uint32_t)fabric_mux_termination_signal_address << "\n";
    DPRINT << "num mux clients: " << (uint32_t)num_mux_clients << "\n";

    const uint32_t receiver_base_address = get_arg_val<uint32_t>(0);
    const uint32_t receive_semaphore_addr = get_arg_val<uint32_t>(1);
    const uint32_t core_noc_x = get_arg_val<uint32_t>(2);
    const uint32_t core_noc_y = get_arg_val<uint32_t>(3);
    const uint32_t page_idx_start = 0;
    const uint32_t page_idx_end = input_num_tiles;
    const uint8_t dst_num_hops = 1;

    const uint32_t max_pages_per_packet_l = input_num_tiles;

    const uint32_t aligned_page_size_bytes = round_up(page_size_bytes, alignment);

    // reusing the last arg for fabric setup, therefore index overlaps.
    size_t arg_idx = 4;
    uint32_t chunk_size = input_num_tiles;

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

    DPRINT << "FABRIC MUX 1 RT ARGS: \n";
    DPRINT << "is termination master: " << (uint32_t)is_termination_master << "\n";
    DPRINT << "fabric mux x: " << (uint32_t)fabric_mux_x << "\n";
    DPRINT << "fabric mux y: " << (uint32_t)fabric_mux_y << "\n";
    DPRINT << "channel base address: " << (uint32_t)fabric_mux_channel_base_address << "\n";
    DPRINT << "connection info address: " << (uint32_t)fabric_mux_connection_info_address << "\n";
    DPRINT << "connection handshake address: " << (uint32_t)fabric_mux_connection_handshake_address << "\n";
    DPRINT << "flow control address: " << (uint32_t)fabric_mux_flow_control_address << "\n";
    DPRINT << "buffer index address: " << (uint32_t)fabric_mux_buffer_index_address << "\n";
    DPRINT << "channel id: " << (uint32_t)fabric_mux_channel_id << "\n";
    DPRINT << "terminaton sync address: " << (uint32_t)termination_sync_address << "\n";
    DPRINT << "local fabric mux status address: " << (uint32_t)local_fabric_mux_status_address << "\n";
    DPRINT << "local flow control address: " << (uint32_t)local_flow_control_address << "\n";
    DPRINT << "local teardown address: " << (uint32_t)local_teardown_address << "\n";
    DPRINT << "local buffer index address: " << (uint32_t)local_buffer_index_address << "\n";
    DPRINT << "termination master noc x: " << (uint32_t)termination_master_noc_x << "\n";
    DPRINT << "termination master noc y: " << (uint32_t)termination_master_noc_y << "\n";

    DPRINT << "is termination master: " << (uint32_t)is_termination_master << "\n";

    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* mux_connection_handle;
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> mux_connection;
    DPRINT << "before building connection to fabric endpoint\n";
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
    DPRINT << "after building connection to fabric endpoint\n";
    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);

    DPRINT << "after waiting for fabric endpoint ready\n";
    tt::tt_fabric::fabric_client_connect(*mux_connection_handle);

    // set up packet header buffer
    cb_reserve_back(packet_header_cb_id, 1);
    uint32_t packet_header_addr = get_read_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    auto* packet_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)packet_header_ptr, dst_num_hops);

    const uint32_t new_payload_size_bytes =
        payload_size_bytes + 2 * aligned_page_size_bytes;  // add the extra size for s and m

    // working memory to hold coalesced packet
    DPRINT << "before reserving back packet cb\n";
    DPRINT << "the packet cb id: " << (uint32_t)packet_cb_id << "\n";
    cb_reserve_back(packet_cb_id, 1);
    const uint32_t packet_base_addr = get_write_ptr(packet_cb_id);
    // cb_push_back(packet_cb_id, 1);

    // initial packet size
    uint32_t curr_pages_per_packet = std::min(max_pages_per_packet_l, page_idx_end - page_idx_start);
    uint32_t packet_idx = 0;

    DPRINT << "before noc semaphore wait\n";
    // wait for receiver to signal it is ready
    auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive_semaphore_addr);
    noc_semaphore_wait_min(local_semaphore_ptr, 1);
    // clean up semaphore – needs to be done before the sender side semaphore increment if we're re-using the semaphore
    // in subsequent program cache hits
    noc_semaphore_set(local_semaphore_ptr, 0);
    DPRINT << "after noc semaphore set\n";

    DPRINT << "writer waiting front on cbs: l, s, m\n";
    DPRINT << "cb id l: " << (uint32_t)cb_id_l << "\n";
    DPRINT << "cb id s: " << (uint32_t)cb_id_s << "\n";
    DPRINT << "cb id m: " << (uint32_t)cb_id_m << "\n";
    cb_wait_front(cb_id_l, chunk_size);
    DPRINT << "printing output of compute from cb_id_l\n";
    print_full_tile(cb_id_l, 10, false);
    uint32_t src_page_base_addr = get_read_ptr(cb_id_l);
    tt_memmove<false, false, false, 0>(packet_base_addr, src_page_base_addr, payload_size_bytes);
    cb_pop_front(cb_id_l, chunk_size);

    cb_wait_front(cb_id_s, 1);
    DPRINT << "printing output of compute from cb_id_s\n";
    print_full_tile(cb_id_s, 0, false);
    const uint32_t src_page_base_addr_s = get_read_ptr(cb_id_s);
    tt_memmove<false, false, false, 0>(
        packet_base_addr + payload_size_bytes, src_page_base_addr_s, aligned_page_size_bytes);
    cb_pop_front(cb_id_s, 1);

    cb_wait_front(cb_id_m, 1);
    DPRINT << "printing output of compute from cb_id_m\n";
    print_full_tile(cb_id_m, 0, false);
    const uint32_t src_page_base_addr_m = get_read_ptr(cb_id_m);
    tt_memmove<false, false, false, 0>(
        packet_base_addr + payload_size_bytes + aligned_page_size_bytes, src_page_base_addr_m, aligned_page_size_bytes);
    cb_pop_front(cb_id_m, 1);
    DPRINT << "before sending packet to root device 1\n";

    DPRINT << "print data from packet cb before send\n";
    print_full_tile(packet_cb_id, 5, true);
    cb_push_back(packet_cb_id, 1);

    // const uint64_t dst_noc_addr = dst_buffer.get_noc_addr(packet_idx, 0, 0);
    const uint64_t dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, receiver_base_address);
    packet_header_ptr->to_noc_unicast_write(
        NocUnicastCommandHeader{dst_noc_addr}, align(new_payload_size_bytes, alignment));

    // perform_payload_send(mux_connection, packet_base_addr, new_payload_size_bytes, packet_header_ptr);
    mux_connection.wait_for_empty_write_slot();
    mux_connection.send_payload_without_header_non_blocking_from_address(packet_base_addr, new_payload_size_bytes);
    mux_connection.send_payload_flush_non_blocking_from_address(
        (uint32_t)packet_header_ptr, sizeof(PACKET_HEADER_TYPE));

    cb_reserve_back(packet_header_cb_id, 1);
    const uint32_t sem_header_addr = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    const uint64_t receive_sem_noc_addr = get_noc_addr(receive_semaphore_addr);

    auto* sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(sem_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)sem_header_ptr, dst_num_hops);
    sem_header_ptr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{receive_sem_noc_addr, 1});

    mux_connection.wait_for_empty_write_slot();
    mux_connection.send_payload_flush_blocking_from_address((uint32_t)sem_header_ptr, packet_header_size_bytes);

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
    DPRINT << "root2 writer kernels completed\n";
}
