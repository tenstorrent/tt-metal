// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "api/dataflow/dataflow_api.h"
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

inline void write_data(
    uint32_t dst_addr_l,
    uint32_t dst_addr_s,
    uint32_t dst_addr_m,
    uint32_t page_bytes,
    uint32_t core_noc_x,
    uint32_t core_noc_y,
    uint32_t cb_int_cb_l,
    uint32_t cb_int_cb_s,
    uint32_t cb_int_cb_m,
    uint32_t onetile,
    uint32_t input_num_tiles) {
    cb_wait_front(cb_int_cb_l, input_num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_int_cb_l);
    uint64_t dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dst_addr_l, 0);
    noc_async_write(l1_read_addr, dst_noc_addr, input_num_tiles * page_bytes);
    cb_pop_front(cb_int_cb_l, input_num_tiles);

    //  for tensor s
    cb_wait_front(cb_int_cb_s, onetile);
    l1_read_addr = get_read_ptr(cb_int_cb_s);
    dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dst_addr_s, 0);
    noc_async_write(l1_read_addr, dst_noc_addr, onetile * page_bytes);
    cb_pop_front(cb_int_cb_s, onetile);

    // for tensor m
    cb_wait_front(cb_int_cb_m, onetile);
    l1_read_addr = get_read_ptr(cb_int_cb_m);
    dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dst_addr_m, 0);
    noc_async_write(l1_read_addr, dst_noc_addr, onetile * page_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_int_cb_m, onetile);
}

void kernel_main() {
    DPRINT << " start of writer 1 kernel\n";
    constexpr uint32_t fabric_ct_idx = get_compile_time_arg_val(0);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t alignment = get_compile_time_arg_val(3);
    constexpr uint32_t input_num_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t payload_size_bytes = get_compile_time_arg_val(6);

    constexpr uint32_t cb_int_cb_l = get_compile_time_arg_val(7);
    constexpr uint32_t cb_int_cb_s = get_compile_time_arg_val(8);
    constexpr uint32_t cb_int_cb_m = get_compile_time_arg_val(9);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(fabric_ct_idx);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(fabric_ct_idx + 1);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(fabric_ct_idx + 2);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(fabric_ct_idx + 3);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(fabric_ct_idx + 4);

    // ROUND 1: send data to neighbor
    DPRINT << "round1\n";

    size_t arg_idx = 0;
    const uint32_t receiver_base_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t receive_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t current_core_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t current_core_noc_y = get_arg_val<uint32_t>(arg_idx++);
    // for round 2
    uint32_t final_dst_addr_l = get_arg_val<uint32_t>(arg_idx++);
    uint32_t final_dst_addr_s = get_arg_val<uint32_t>(arg_idx++);
    uint32_t final_dst_addr_m = get_arg_val<uint32_t>(arg_idx++);

    const uint8_t dst_num_hops = 1;

    const uint32_t aligned_page_size_bytes = round_up(page_size_bytes, alignment);

    const uint32_t new_payload_size_bytes =
        payload_size_bytes + 2 * aligned_page_size_bytes;  // add the extra size for s and m

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
    DPRINT << "after wait for fabric endpoint ready\n";

    tt::tt_fabric::fabric_client_connect(*mux_connection_handle);
    DPRINT << "after fabric client connect\n";

    cb_reserve_back(packet_header_cb_id, 1);
    uint32_t packet_header_addr = get_read_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);
    DPRINT << "after reserving packet header buffer\n";

    auto* packet_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)packet_header_ptr, dst_num_hops);

    DPRINT << "AFTER fabric_set_unicast_route\n";
    //  wait for receiver to signal it is ready
    DPRINT << "waiting for barrier semaphore at address: " << (uint32_t)receive_semaphore_addr << "\n";
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive_semaphore_addr), 1);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receive_semaphore_addr), 0);
    DPRINT << "after receiving ready signal from receiver\n";

    cb_wait_front(packet_cb_id, 1);
    DPRINT << "after reserving packet buffer\n";
    uint32_t packet_base_addr = get_write_ptr(packet_cb_id);

    DPRINT << "sending fabric sem at addr: " << (uint32_t)receive_semaphore_addr << "\n";
    const uint64_t dst_noc_addr = get_noc_addr(current_core_noc_x, current_core_noc_y, receiver_base_address);
    const uint64_t receive_sem_noc_addr = get_noc_addr(current_core_noc_x, current_core_noc_y, receive_semaphore_addr);

    // Use fused packet API to send data + semaphore increment in a single packet
    packet_header_ptr->to_noc_fused_unicast_write_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, receive_sem_noc_addr, 1, true},
        align(new_payload_size_bytes, alignment));

    mux_connection.wait_for_empty_write_slot();
    DPRINT << "after wait for empty write slot\n";
    mux_connection.send_payload_without_header_non_blocking_from_address(packet_base_addr, new_payload_size_bytes);
    DPRINT << "after sending payload to receiver\n";
    mux_connection.send_payload_flush_blocking_from_address((uint32_t)packet_header_ptr, sizeof(PACKET_HEADER_TYPE));
    DPRINT << "after sending data to receiver\n";

    cb_pop_front(packet_cb_id, 1);

    tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle);
    DPRINT << "after fabric_client_disconnect\n";

    // Writer1 uses forward mux - signal the forward mux termination master (Reader2)
    {
        uint64_t dest_addr =
            safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
        noc_semaphore_inc(dest_addr, 1);
        noc_async_atomic_barrier();
        DPRINT << "signaled forward mux termination master\n";
    }

    DPRINT << "end of round 1\n";

    DPRINT << "round2\n";
    // ROUND 2: wait for compute and write output
    // receives l, m, s tensors from the compute kernel and writes to the final output buffers
    constexpr uint32_t onetile = 1;
    write_data(
        final_dst_addr_l,
        final_dst_addr_s,
        final_dst_addr_m,
        page_size_bytes,
        core_noc_x,
        core_noc_y,
        cb_int_cb_l,
        cb_int_cb_s,
        cb_int_cb_m,
        onetile,
        input_num_tiles);
    DPRINT << "end of writer 1 kernel\n";
}
