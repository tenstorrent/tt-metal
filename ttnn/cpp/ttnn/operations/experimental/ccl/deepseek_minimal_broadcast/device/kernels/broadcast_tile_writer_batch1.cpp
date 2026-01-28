// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"

using address_t = uint32_t;
using namespace tt::tt_fabric::linear::experimental;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(1);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(2);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(3);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(4);
constexpr bool is_sender = get_compile_time_arg_val(5);
constexpr uint32_t core_noc_x = get_compile_time_arg_val(6);
constexpr uint32_t core_noc_y = get_compile_time_arg_val(7);
constexpr bool is_secondary_sender = get_compile_time_arg_val(8);
constexpr bool has_secondary_target = get_compile_time_arg_val(9);
constexpr bool has_reverse_secondary_connection = get_compile_time_arg_val(10);
constexpr uint32_t start_distance_in_hops_forward = get_compile_time_arg_val(11);
constexpr uint32_t range_hops_forward = get_compile_time_arg_val(12);
constexpr uint32_t start_distance_in_hops_backward = get_compile_time_arg_val(13);
constexpr uint32_t range_hops_backward = get_compile_time_arg_val(14);
constexpr bool using_persistent_buffers = get_compile_time_arg_val(15);

// Number of primary axis connections (forward + backward)
constexpr uint32_t num_primary_connections =
    (start_distance_in_hops_forward > 0 ? 1 : 0) + (start_distance_in_hops_backward > 0 ? 1 : 0);

// Index of the secondary axis connection
constexpr uint32_t secondary_connection_idx = num_primary_connections;

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    uint32_t tensor_address0 = get_arg_val<uint32_t>(arg_idx++);
    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tile_id_end = get_arg_val<uint32_t>(arg_idx++);
    bool wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    bool reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ring_index = get_arg_val<uint32_t>(arg_idx++);
    const size_t secondary_sync_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_connections = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;

    auto sem_route_id = PacketHeaderPool::allocate_header_n(num_primary_connections);
    auto fused_route_id = PacketHeaderPool::allocate_header_n(num_primary_connections);
    // Allocate separate route for secondary axis unicast (if applicable)
    auto secondary_route_id = has_secondary_target ? PacketHeaderPool::allocate_header_n(1) : 0;
    // Allocate route for reverse secondary connection (secondary sender -> primary sender)
    auto reverse_secondary_route_id = has_reverse_secondary_connection ? PacketHeaderPool::allocate_header_n(1) : 0;
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;

    open_connections(fabric_connection, num_connections, arg_for_fab);

    uint8_t starts[] = {
        static_cast<uint8_t>(start_distance_in_hops_forward), static_cast<uint8_t>(start_distance_in_hops_backward)};
    uint8_t ranges[] = {static_cast<uint8_t>(range_hops_forward), static_cast<uint8_t>(range_hops_backward)};
    if (ranges[0] == 0) {
        starts[0] = starts[1];
        ranges[0] = ranges[1];
    }

    // Configure fused route for payload + semaphore increment
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader fused_header(0, 0, 1, true);
    fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<
        UnicastFusedAtomicIncUpdateMask::Val | UnicastFusedAtomicIncUpdateMask::Flush>(
        fabric_connection, fused_route_id, starts, ranges, fused_header, tensor0_page_size);

    uint32_t num_total_targets = num_targets_forward_direction + num_targets_backward_direction;

    uint64_t barrier_sem_noc_addr_in_pkt = safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
    uint64_t secondary_sync_sem_noc_addr_in_pkt =
        safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, secondary_sync_sem, 0);

    // Secondary axis barrier between primary sender and secondary sender
    // Both devices send an atomic inc to each other and wait
    if constexpr (!using_persistent_buffers) {
        if constexpr (has_secondary_target || has_reverse_secondary_connection) {
            auto& secondary_slot = fabric_connection.get(secondary_connection_idx);
            auto route_id = has_secondary_target ? secondary_route_id : reverse_secondary_route_id;
            volatile PACKET_HEADER_TYPE* sec_sync_header = PacketHeaderPool::header_table[route_id].first;
            fabric_set_unicast_route(fabric_connection, sec_sync_header, secondary_connection_idx);

            // Send atomic inc to the other device's secondary sync semaphore
            fabric_unicast_noc_unicast_atomic_inc(
                &secondary_slot.sender,
                sec_sync_header,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{secondary_sync_sem_noc_addr_in_pkt, 1},
                1);

            // Wait for the other device's atomic inc on secondary_sync_sem
            volatile tt_l1_ptr uint32_t* sync_sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(secondary_sync_sem);
            noc_semaphore_wait_min(sync_sem_ptr, 1);
            noc_semaphore_set(sync_sem_ptr, 0);
        }

        if (!is_sender) {
            // Now do the column-wise barrier (primary axis sync)
            // do it for the sender later to overlap with data sending
            fabric_multicast_noc_unicast_atomic_inc_set_state<
                UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                fabric_connection,
                sem_route_id,
                starts,
                ranges,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, static_cast<uint32_t>(1)});
            fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_connection,
                sem_route_id,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), num_total_targets);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
        }
    }

    if (is_sender) {
        uint32_t num_pages_to_read = std::min(tile_id_end - tile_id_start, packet_size_in_pages);
        cb_wait_front(cb0_id, packet_size_in_pages);
        size_t l1_read_addr = get_read_ptr(cb0_id);
        uint64_t dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, tensor_address0, 0);
        dst_noc_addr += tile_id_start * tensor0_page_size;

        uint64_t out_ready_sem_noc_addr_in_pkt =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);

        // For dual-axis mode: first unicast to secondary sender, then mcast along primary axis
        if constexpr (has_secondary_target) {
            auto& secondary_slot = fabric_connection.get(secondary_connection_idx);
            volatile PACKET_HEADER_TYPE* secondary_header = PacketHeaderPool::header_table[secondary_route_id].first;

            // Set up unicast route for 2D fabric
            fabric_set_unicast_route(fabric_connection, secondary_header, secondary_connection_idx);

            // Send data + semaphore increment to secondary sender
            fabric_unicast_noc_fused_unicast_with_atomic_inc(
                &secondary_slot.sender,
                secondary_header,
                l1_read_addr,
                tensor0_page_size * num_pages_to_read,
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    dst_noc_addr, out_ready_sem_noc_addr_in_pkt, 1, true},
                1);  // 1 hop to secondary sender
        }
        if constexpr (!using_persistent_buffers) {
            // Now do the column-wise barrier (primary axis sync)
            fabric_multicast_noc_unicast_atomic_inc_set_state<
                UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                fabric_connection,
                sem_route_id,
                starts,
                ranges,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, static_cast<uint32_t>(1)});
            fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                fabric_connection,
                sem_route_id,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), num_total_targets);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
        }
        fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<
            UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr |
            UnicastFusedAtomicIncUpdateMask::PayloadSize>(
            fabric_connection,
            fused_route_id,
            l1_read_addr,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, out_ready_sem_noc_addr_in_pkt, 1, true},
            tensor0_page_size * num_pages_to_read);

        noc_async_write(l1_read_addr, dst_noc_addr, tensor0_page_size * num_pages_to_read);

        // increment locally
        uint64_t out_ready_sem_noc_addr =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
        noc_semaphore_inc(out_ready_sem_noc_addr, 1);

        // 3. wait for mcast output ready semaphore
        if (wait_output_semaphore) {
            volatile tt_l1_ptr uint32_t* sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr);
            noc_semaphore_wait(sem_ptr, out_ready_sem_wait_value);
        }

        // 4. global semaphore reset
        if (reset_global_semaphore) {
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr), 0);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb0_id, packet_size_in_pages);

    } else if constexpr (is_secondary_sender) {
        // Secondary sender: wait for data from primary sender, then broadcast along primary axis
        // First wait for data to arrive from primary sender
        if (wait_output_semaphore) {
            volatile tt_l1_ptr uint32_t* sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr);
            noc_semaphore_wait(sem_ptr, out_ready_sem_wait_value);
        }

        // Reset semaphore after receiving data
        if (reset_global_semaphore) {
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr), 0);
        }

        // broadcast the received data along the primary axis
        uint32_t num_pages_to_read = std::min(tile_id_end - tile_id_start, packet_size_in_pages);
        uint64_t src_noc_addr = get_noc_addr(core_noc_x, core_noc_y, tensor_address0, 0);
        src_noc_addr += tile_id_start * tensor0_page_size;

        // Mcast the data along primary axis
        uint64_t out_ready_sem_noc_addr_in_pkt =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);

        fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<
            UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr |
            UnicastFusedAtomicIncUpdateMask::PayloadSize>(
            fabric_connection,
            fused_route_id,
            static_cast<size_t>(tensor_address0 + tile_id_start * tensor0_page_size),
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{src_noc_addr, out_ready_sem_noc_addr_in_pkt, 1, true},
            tensor0_page_size * num_pages_to_read);
        noc_async_writes_flushed();

    } else {
        // Receiver: wait for data from broadcaster
        if (wait_output_semaphore) {
            volatile tt_l1_ptr uint32_t* sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr);
            noc_semaphore_wait(sem_ptr, out_ready_sem_wait_value);
        }

        // Reset global semaphore
        if (reset_global_semaphore) {
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr), 0);
        }
    }
    close_connections(fabric_connection);

    noc_async_write_barrier();
}
