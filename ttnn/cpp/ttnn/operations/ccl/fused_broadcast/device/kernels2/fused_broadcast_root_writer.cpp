// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "ttnn/operations/point_to_point/device/kernels/dataflow/common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/heterogeneous_data_structs.hpp"

using tt::data_movement::common::round_up;
using tt::data_movement::common::tt_memmove;
using address_t = uint32_t;
using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS (Combined P2P + Broadcast)
    ///////////////////////////////////////////////////
    constexpr uint32_t sender_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t alignment = get_compile_time_arg_val(3);

    // Broadcast specific compile-time args
    constexpr uint32_t page_size = get_compile_time_arg_val(4);
    constexpr uint32_t row_size = get_compile_time_arg_val(5);
    constexpr uint32_t max_packet_size = get_compile_time_arg_val(6);
    constexpr uint32_t num_rows_per_packet = get_compile_time_arg_val(7);
    constexpr uint32_t num_packets_per_row = get_compile_time_arg_val(8);
    constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(9);
    constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(10);
    constexpr uint32_t start_distance_in_hops_forward = get_compile_time_arg_val(11);
    constexpr uint32_t range_hops_forward = get_compile_time_arg_val(12);
    constexpr uint32_t start_distance_in_hops_backward = get_compile_time_arg_val(13);
    constexpr uint32_t range_hops_backward = get_compile_time_arg_val(14);
    constexpr auto dst_buffer_args = TensorAccessorArgs<15>();

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    // Debug: Print compile-time args
    DPRINT << "ROOT WRITER: sender_cb_id=" << (uint32_t)sender_cb_id
           << ", packet_header_cb_id=" << (uint32_t)packet_header_cb_id << ", packet_cb_id=" << (uint32_t)packet_cb_id
           << ENDL();
    DPRINT << "ROOT WRITER: alignment=" << (uint32_t)alignment << ", page_size=" << (uint32_t)page_size
           << ", row_size=" << (uint32_t)row_size << ENDL();
    DPRINT << "ROOT WRITER: max_packet_size=" << (uint32_t)max_packet_size
           << ", num_rows_per_packet=" << (uint32_t)num_rows_per_packet
           << ", num_packets_per_row=" << (uint32_t)num_packets_per_row << ENDL();
    DPRINT << "ROOT WRITER: num_targets_forward_direction=" << (uint32_t)num_targets_forward_direction
           << ", num_targets_backward_direction=" << (uint32_t)num_targets_backward_direction << ENDL();
    DPRINT << "ROOT WRITER: start_distance_in_hops_forward=" << (uint32_t)start_distance_in_hops_forward
           << ", range_hops_forward=" << (uint32_t)range_hops_forward << ENDL();
    DPRINT << "ROOT WRITER: start_distance_in_hops_backward=" << (uint32_t)start_distance_in_hops_backward
           << ", range_hops_backward=" << (uint32_t)range_hops_backward << ENDL();

    ///////////////////////////////////////////////////
    // RUNTIME ARGS (Combined P2P + Broadcast)
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;

    // SP fabric arg start index (passed from program factory)
    const size_t sp_fabric_arg_start = get_arg_val<uint32_t>(arg_idx++);

    // P2P Phase arguments (TP horizontal)
    const uint32_t tp_receiver_base_address = get_arg_val<uint32_t>(arg_idx++);
    const auto tp_page_idx_start = get_arg_val<uint32_t>(arg_idx++);
    const auto tp_page_idx_end = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t tp_dst_num_hops = get_arg_val<uint32_t>(arg_idx++);
    const auto tp_page_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const auto tp_payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const auto tp_max_pages_per_packet = get_arg_val<uint32_t>(arg_idx++);
    const auto tp_page_segments = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tp_receive_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const bool tp_dst_is_forward = get_arg_val<uint32_t>(arg_idx++);

    // Debug: Print P2P runtime args
    DPRINT << "ROOT WRITER: tp_receiver_base_address=" << (uint32_t)tp_receiver_base_address
           << ", tp_page_idx_start=" << (uint32_t)tp_page_idx_start << ", tp_page_idx_end=" << (uint32_t)tp_page_idx_end
           << ENDL();
    DPRINT << "ROOT WRITER: tp_dst_num_hops=" << (uint32_t)tp_dst_num_hops
           << ", tp_page_size_bytes=" << (uint32_t)tp_page_size_bytes
           << ", tp_payload_size_bytes=" << (uint32_t)tp_payload_size_bytes << ENDL();
    DPRINT << "ROOT WRITER: tp_max_pages_per_packet=" << (uint32_t)tp_max_pages_per_packet
           << ", tp_page_segments=" << (uint32_t)tp_page_segments
           << ", tp_receive_semaphore_addr=" << (uint32_t)tp_receive_semaphore_addr << ENDL();
    DPRINT << "ROOT WRITER: tp_dst_is_forward=" << (uint32_t)tp_dst_is_forward << ENDL();

    // SP Broadcast Phase arguments (column vertical)
    const uint32_t sp_tensor_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sp_out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sp_row_id_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sp_row_id_end = get_arg_val<uint32_t>(arg_idx++);
    const bool sp_wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const bool sp_reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t sp_out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t sp_out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sp_out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sp_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t sp_barrier_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t sp_barrier_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sp_num_connections = get_arg_val<uint32_t>(arg_idx++);

    // Debug: Print SP Broadcast runtime args
    DPRINT << "ROOT WRITER: sp_tensor_address=" << (uint32_t)sp_tensor_address
           << ", sp_out_ready_sem_bank_addr=" << (uint32_t)sp_out_ready_sem_bank_addr << ENDL();
    DPRINT << "ROOT WRITER: sp_row_id_start=" << (uint32_t)sp_row_id_start
           << ", sp_row_id_end=" << (uint32_t)sp_row_id_end << ENDL();
    DPRINT << "ROOT WRITER: sp_wait_output_semaphore=" << (uint32_t)sp_wait_output_semaphore
           << ", sp_reset_global_semaphore=" << (uint32_t)sp_reset_global_semaphore << ENDL();
    DPRINT << "ROOT WRITER: sp_out_ready_sem_noc0_x=" << (uint32_t)sp_out_ready_sem_noc0_x
           << ", sp_out_ready_sem_noc0_y=" << (uint32_t)sp_out_ready_sem_noc0_y
           << ", sp_out_ready_sem_wait_value=" << (uint32_t)sp_out_ready_sem_wait_value << ENDL();
    DPRINT << "ROOT WRITER: sp_barrier_sem=" << (uint32_t)sp_barrier_sem
           << ", sp_barrier_sem_noc0_x=" << (uint32_t)sp_barrier_sem_noc0_x
           << ", sp_barrier_sem_noc0_y=" << (uint32_t)sp_barrier_sem_noc0_y << ENDL();
    DPRINT << "ROOT WRITER: sp_num_connections=" << (uint32_t)sp_num_connections << ENDL();

    // SP broadcast fabric args (starting from sp_fabric_arg_start)

    size_t tp_fabric_arg_start = arg_idx;

    ///////////////////////////////////////////////////
    // PHASE 1: TP P2P (Horizontal) - Send to TP partner
    ///////////////////////////////////////////////////
    const uint32_t tp_aligned_page_size_bytes = round_up(tp_page_size_bytes, alignment);

    // Setup P2P fabric connection for TP phase
    auto tp_fabric_connection = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(tp_fabric_arg_start);

    // Set up TP P2P packet header buffer
    cb_reserve_back(packet_header_cb_id, 1);
    uint32_t tp_packet_header_addr = get_read_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    auto* tp_packet_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(tp_packet_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)tp_packet_header_ptr, tp_dst_num_hops);

    const auto tp_dst_buffer = TensorAccessor(dst_buffer_args, tp_receiver_base_address, tp_payload_size_bytes);

    // TP P2P working memory to hold coalesced packet
    cb_reserve_back(packet_cb_id, 1);
    const uint32_t tp_packet_base_addr = get_write_ptr(packet_cb_id);
    cb_push_back(packet_cb_id, 1);

    // TP P2P initial packet size
    uint32_t tp_curr_pages_per_packet = std::min(tp_max_pages_per_packet, tp_page_idx_end - tp_page_idx_start);
    uint32_t tp_packet_idx = tp_page_idx_start / tp_max_pages_per_packet;

    // Wait for TP receiver to signal it is ready
    auto tp_local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tp_receive_semaphore_addr);
    noc_semaphore_wait(tp_local_semaphore_ptr, 1);
    noc_semaphore_set(tp_local_semaphore_ptr, 0);

    tp_fabric_connection.open_finish();
    auto& tp_connection_direction =
        tt::point_to_point::common::connection_direction_collection(tp_dst_is_forward, tp_fabric_connection);

    // Allocate intermediate buffer in L1 to store data for both phases
    const uint32_t total_data_size = (tp_page_idx_end - tp_page_idx_start) * tp_page_size_bytes;
    uint32_t intermediate_buffer_addr = 0;  // Will be set when we read first data

    // TP P2P main processing loop - send data horizontally to TP partner AND store for SP phase
    uint32_t stored_data_offset = 0;
    for (uint32_t page_idx = tp_page_idx_start, packet_page_idx = 0; page_idx < tp_page_idx_end; ++page_idx) {
        cb_wait_front(sender_cb_id, 1);
        const uint32_t src_page_base_addr = get_read_ptr(sender_cb_id);

        // Set intermediate buffer address on first iteration
        if (page_idx == tp_page_idx_start) {
            // Use area after packet buffer for intermediate storage
            intermediate_buffer_addr = tp_packet_base_addr + (tp_max_pages_per_packet * tp_aligned_page_size_bytes);
        }

        for (uint32_t page_segment_idx = 0; page_segment_idx < tp_page_segments; ++page_segment_idx) {
            const uint32_t page_offset = page_segment_idx * tp_payload_size_bytes;
            const uint32_t src_addr = src_page_base_addr + page_offset;
            const uint32_t transfer_size_bytes = std::min(tp_page_size_bytes - page_offset, tp_payload_size_bytes);

            // Copy page to TP packet buffer with offset
            const uint32_t packet_addr = tp_packet_base_addr + packet_page_idx * tp_aligned_page_size_bytes;
            tt_memmove<false, false, false, 0>(packet_addr, src_addr, transfer_size_bytes);

            // ALSO copy to intermediate buffer for SP broadcast phase
            const uint32_t intermediate_addr = intermediate_buffer_addr + stored_data_offset;
            tt_memmove<false, false, false, 0>(intermediate_addr, src_addr, transfer_size_bytes);
            stored_data_offset += transfer_size_bytes;

            ++packet_page_idx;
            if (packet_page_idx >= tp_curr_pages_per_packet) {
                const uint64_t dst_noc_addr = get_noc_addr(tp_packet_idx, tp_dst_buffer, 0, 0);
                tt::tt_fabric::linear::to_noc_unicast_write(
                    align(tp_payload_size_bytes, alignment), tp_packet_header_ptr, tp_packet_idx, tp_dst_buffer);
                perform_payload_send(
                    tp_connection_direction, tp_packet_base_addr, tp_payload_size_bytes, tp_packet_header_ptr);

                // Reset TP counters
                packet_page_idx = 0;
                tp_curr_pages_per_packet = std::min(tp_max_pages_per_packet, tp_page_idx_end - page_idx - 1);
                ++tp_packet_idx;
            }
        }
        cb_pop_front(sender_cb_id, 1);
    }

    // TP P2P completion - signal TP receiver
    cb_reserve_back(packet_header_cb_id, 1);
    const uint32_t tp_sem_header_addr = get_write_ptr(packet_header_cb_id);
    cb_push_back(packet_header_cb_id, 1);

    const uint64_t tp_receive_sem_noc_addr = get_noc_addr(tp_receive_semaphore_addr);

    auto* tp_sem_header_ptr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(tp_sem_header_addr);
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)tp_sem_header_ptr, tp_dst_num_hops);
    tp_sem_header_ptr->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{tp_receive_sem_noc_addr, 1});

    tp_connection_direction.wait_for_empty_write_slot();
    tp_connection_direction.send_payload_flush_blocking_from_address(
        (uint32_t)tp_sem_header_ptr, packet_header_size_bytes);

    tp_fabric_connection.close();

    ///////////////////////////////////////////////////
    // PHASE 2: SP Broadcast (Vertical) - Broadcast down column
    ///////////////////////////////////////////////////

    // SP Broadcast fabric setup
    auto sp_unicast_route_id = PacketHeaderPool::allocate_header_n(sp_num_connections);
    auto sp_scatter_route_id = PacketHeaderPool::allocate_header_n(sp_num_connections);
    auto sp_sem_route_id = PacketHeaderPool::allocate_header_n(sp_num_connections);
    tt::tt_fabric::RoutingPlaneConnectionManager sp_fabric_connection;
    // SP Broadcast tensor setup
#ifdef SHARDED
    typedef ShardedInfo<
        get_compile_time_arg_val(20),
        get_compile_time_arg_val(21),
        get_compile_time_arg_val(22),
        get_compile_time_arg_val(23),
        get_compile_time_arg_val(24),
        get_compile_time_arg_val(25),
        get_compile_time_arg_val(26)>
        sp_tensor_shard_info;
    const auto [sp_mapping_table, sp_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<sp_tensor_shard_info>(get_arg_addr(sp_fabric_arg_start));
    experimental::ShardedAddrGen<sp_tensor_shard_info> sp_tensor_addrgen = {
        .bank_base_address = sp_tensor_address, .shard_array = sp_mapping_table};
    size_t sp_fab_idx = sp_fabric_arg_start + sp_rt_increment;
    open_connections(sp_fabric_connection, sp_num_connections, sp_fab_idx);
#else
    constexpr auto sp_tensor_args = TensorAccessorArgs<20>();
    auto sp_tensor_addrgen = TensorAccessor(sp_tensor_args, sp_tensor_address, page_size);
    open_connections(sp_fabric_connection, sp_num_connections, sp_fabric_arg_start);
#endif
    // SP Broadcast routing setup (vertical - down column)
    uint8_t sp_starts[] = {
        static_cast<uint8_t>(start_distance_in_hops_forward), static_cast<uint8_t>(start_distance_in_hops_backward)};
    uint8_t sp_ranges[] = {static_cast<uint8_t>(range_hops_forward), static_cast<uint8_t>(range_hops_backward)};
    if (sp_ranges[0] == 0) {
        sp_starts[0] = sp_starts[1];
        sp_ranges[0] = sp_ranges[1];
    }

    // Setup SP broadcast multicast routing
    fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        sp_fabric_connection, sp_unicast_route_id, sp_starts, sp_ranges, nullptr, page_size);
    fabric_multicast_noc_scatter_write_set_state<
        UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
        sp_fabric_connection,
        sp_scatter_route_id,
        sp_starts,
        sp_ranges,
        NocUnicastScatterCommandHeader{
            {0, 0},  // ignore
            static_cast<uint16_t>(page_size)},
        page_size * 2);

    // SP Broadcast synchronization
    uint32_t sp_num_total_targets = num_targets_forward_direction + num_targets_backward_direction;

    // Barrier semaphore with devices down the same column
    uint64_t sp_barrier_sem_noc_addr_in_pkt =
        safe_get_noc_addr(sp_barrier_sem_noc0_x, sp_barrier_sem_noc0_y, sp_barrier_sem, 0);
    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        sp_fabric_connection,
        sp_sem_route_id,
        sp_starts,
        sp_ranges,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,  // ignore
            static_cast<uint32_t>(1)});
    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        sp_fabric_connection,
        sp_sem_route_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sp_barrier_sem_noc_addr_in_pkt, 0});
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sp_barrier_sem), sp_num_total_targets);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sp_barrier_sem), 0);

    // SP Broadcast main processing loop - broadcast data vertically down column
    // Use the intermediate buffer that was filled during TP P2P phase
    uint32_t sp_row_id = sp_row_id_start;
    size_t l1_read_addr = intermediate_buffer_addr;  // Start from stored data

    while (sp_row_id < sp_row_id_end) {
        uint32_t num_rows_read = 0;
        uint32_t num_rows_to_read = std::min(sp_row_id_end - sp_row_id, num_rows_per_packet);
        while (num_rows_read < num_rows_to_read) {
            // Scatter-write supports up to 2 distinct addresses
            uint32_t num_rows_for_current_packet = std::min<uint32_t>(num_rows_to_read - num_rows_read, 2);
            if (num_rows_for_current_packet == 1) {
                noc_async_write(l1_read_addr, sp_tensor_addrgen.get_noc_addr(sp_row_id, 0), page_size);
                fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                    sp_fabric_connection,
                    sp_unicast_route_id,
                    l1_read_addr,
                    tt::tt_fabric::NocUnicastCommandHeader{
                        linear::addrgen_detail::get_noc_address(sp_tensor_addrgen, sp_row_id, 0)});
                noc_async_writes_flushed();
                l1_read_addr += page_size;
                sp_row_id++;
                num_rows_read++;
            } else if (num_rows_for_current_packet == 2) {
                noc_async_write(l1_read_addr, sp_tensor_addrgen.get_noc_addr(sp_row_id, 0), page_size);
                noc_async_write(l1_read_addr + page_size, sp_tensor_addrgen.get_noc_addr(sp_row_id + 1, 0), page_size);
                fabric_multicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                    sp_fabric_connection,
                    sp_scatter_route_id,
                    l1_read_addr,
                    tt::tt_fabric::NocUnicastScatterCommandHeader{
                        {linear::addrgen_detail::get_noc_address(sp_tensor_addrgen, sp_row_id, 0),
                         linear::addrgen_detail::get_noc_address(sp_tensor_addrgen, sp_row_id + 1, 0)},
                        static_cast<uint16_t>(page_size)},
                    page_size * 2);
                noc_async_writes_flushed();
                l1_read_addr += page_size * 2;
                sp_row_id += 2;
                num_rows_read += 2;
            } else {
                ASSERT(false);
            }
        }
        // No cb_pop_front needed since we're using intermediate buffer
    }
    // SP Broadcast completion - signal all column receivers
    uint64_t sp_out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(sp_out_ready_sem_noc0_x, sp_out_ready_sem_noc0_y, sp_out_ready_sem_bank_addr, 0);
    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        sp_fabric_connection,
        sp_sem_route_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sp_out_ready_sem_noc_addr_in_pkt, 0});

    // Increment locally
    uint64_t sp_out_ready_sem_noc_addr =
        safe_get_noc_addr(sp_out_ready_sem_noc0_x, sp_out_ready_sem_noc0_y, sp_out_ready_sem_bank_addr);
    noc_semaphore_inc(sp_out_ready_sem_noc_addr, 1);

    // Wait for SP broadcast output ready semaphore
    if (sp_wait_output_semaphore) {
        volatile tt_l1_ptr uint32_t* sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sp_out_ready_sem_bank_addr);
        noc_semaphore_wait(sem_ptr, sp_out_ready_sem_wait_value);
    }

    // Global semaphore reset for SP
    if (sp_reset_global_semaphore) {
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sp_out_ready_sem_bank_addr), 0);
    }

    close_connections(sp_fabric_connection);
    noc_async_write_barrier();
}
