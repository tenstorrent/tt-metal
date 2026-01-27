// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
// Simplified writer kernel for reduce_to_all operation with RELAY + DIRECT ROUTER optimization.
// This kernel runs on SHARD CORES (data cores).
//
// Relay master connects DIRECTLY to the fabric router.
//
// Relay Master Logic (worker 0 per link):
// 1. Connect directly to fabric router (R1 and R2 directions)
// 2. Prepare and send own packet to fabric router
// 3. Poll relay buffers for packets from relay workers 1, 2, 3
// 4. Forward any ready relay packets to fabric router
//
// Relay Worker Logic (workers 1, 2, 3 per link):
// 1. Prepare packet (same as before)
// 2. NOC write packet to relay master's relay buffer
// 3. Signal relay master via semaphore
// 4. Skip fabric connection entirely
//
// R1/R2 ROUTER DIRECTION:
// The program factory determines which fabric router direction to use for R1 and R2
// based on device position in the ring:
//   - Even devices (0,2): R1=FWD router, R2=BWD router
//   - Odd devices (1,3):  R1=BWD router, R2=FWD router

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/common.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include <cstdint>

using tt::data_movement::common::round_up;
using tt::data_movement::common::tt_memmove;

void kernel_main() {
    // ==========================================================================
    // Compile-time args
    // ==========================================================================

    // Compute parameters (must match compute kernel)
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t vDHt = get_compile_time_arg_val(1);

    // CB IDs for data sources
    [[maybe_unused]] constexpr uint32_t cb_local_l = get_compile_time_arg_val(2);  // UNUSED
    [[maybe_unused]] constexpr uint32_t cb_local_s = get_compile_time_arg_val(3);  // UNUSED
    [[maybe_unused]] constexpr uint32_t cb_local_m = get_compile_time_arg_val(4);  // UNUSED
    constexpr uint32_t cb_r1_result_l = get_compile_time_arg_val(5);
    constexpr uint32_t cb_r1_result_s = get_compile_time_arg_val(6);
    constexpr uint32_t cb_r1_result_m = get_compile_time_arg_val(7);

    // Packet/header CBs
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(9);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(10);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(12);

    // Relay configuration
    constexpr uint32_t num_relay_workers = get_compile_time_arg_val(13);
    [[maybe_unused]] constexpr uint32_t relay_buffer_size = get_compile_time_arg_val(14);
    constexpr uint32_t packet_header_size_bytes = get_compile_time_arg_val(15);

    // Note: Fabric router connection info is read from L1 static memory (MEM_TENSIX_FABRIC_CONNECTIONS_BASE)
    // by WorkerToFabricEdmSender::build_from_args<TENSIX>(), so no compile-time args needed here.

    // Derived constants
    constexpr uint32_t out_tiles = Sq_chunk_t * vDHt;
    constexpr uint32_t payload_size_bytes = out_tiles * page_size_bytes;

    constexpr uint8_t num_hops = 1;
    constexpr uint32_t aligned_page_size = ((page_size_bytes + l1_alignment - 1) / l1_alignment) * l1_alignment;

    // ==========================================================================
    // Runtime args
    // ==========================================================================
    size_t arg_idx = 0;

    // Input tensor source addresses
    const uint32_t src_addr_l = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t src_addr_s = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t src_addr_m = get_arg_val<uint32_t>(arg_idx++);

    // R1 destination
    const uint32_t r1_neighbor_dst_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);

    // R2 destination
    const uint32_t r2_neighbor_dst_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);

    // Local core coordinates
    const uint32_t current_core_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t current_core_y = get_arg_val<uint32_t>(arg_idx++);

    // Relay configuration
    const bool is_relay_master = get_arg_val<uint32_t>(arg_idx++);

    // Relay-specific runtime args (different for master vs worker)
    uint32_t relay_buffer_addrs[3] = {0, 0, 0};
    uint32_t relay_sem_addrs[3] = {0, 0, 0};
    uint32_t relay_master_noc_x = 0;
    uint32_t relay_master_noc_y = 0;
    uint32_t my_relay_buffer_addr = 0;
    uint32_t my_relay_sem_addr = 0;

    if (is_relay_master) {
        relay_buffer_addrs[0] = get_arg_val<uint32_t>(arg_idx++);
        relay_buffer_addrs[1] = get_arg_val<uint32_t>(arg_idx++);
        relay_buffer_addrs[2] = get_arg_val<uint32_t>(arg_idx++);
        relay_sem_addrs[0] = get_arg_val<uint32_t>(arg_idx++);
        relay_sem_addrs[1] = get_arg_val<uint32_t>(arg_idx++);
        relay_sem_addrs[2] = get_arg_val<uint32_t>(arg_idx++);
        // Fabric router connection args are parsed below using build_from_args
    } else {
        relay_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
        relay_master_noc_y = get_arg_val<uint32_t>(arg_idx++);
        my_relay_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
        my_relay_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    }

    // Computed values
    const uint32_t total_payload_size = payload_size_bytes + 2 * aligned_page_size;

    // ==========================================================================
    // RELAY WORKER PATH: Prepare packet and send to relay master
    // ==========================================================================
    if (!is_relay_master) {
        // ======================================================================
        // R1 PHASE: Send local data to relay master
        // ======================================================================
        {
            DeviceZoneScopedN("RELAY-R1-PREPARE");

            // Prepare packet: [header][L][S aligned][M aligned]
            cb_reserve_back(packet_header_cb_id, 1);
            uint32_t header_addr = get_write_ptr(packet_header_cb_id);
            auto* header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_addr);
            fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)header, num_hops);

            // Build fused packet header for R1 destination
            const uint64_t r1_dst_noc = get_noc_addr(current_core_x, current_core_y, r1_neighbor_dst_addr);
            const uint64_t r1_sem_noc = get_noc_addr(current_core_x, current_core_y, r1_neighbor_sem_addr);
            header->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{r1_dst_noc, r1_sem_noc, 1, false},
                align(total_payload_size, l1_alignment));

            cb_push_back(packet_header_cb_id, 1);

            // Pack payload
            cb_reserve_back(packet_cb_id, 1);
            uint32_t packet_addr = get_write_ptr(packet_cb_id);
            tt_memmove<true, false, false, 0>(packet_addr, src_addr_l, payload_size_bytes);
            tt_memmove<true, false, false, 0>(packet_addr + payload_size_bytes, src_addr_s, aligned_page_size);
            tt_memmove<true, false, false, 0>(
                packet_addr + payload_size_bytes + aligned_page_size, src_addr_m, aligned_page_size);
            cb_push_back(packet_cb_id, 1);

            // NOC write to relay master's relay buffer: [header][payload]
            uint64_t relay_buffer_noc = get_noc_addr(relay_master_noc_x, relay_master_noc_y, my_relay_buffer_addr);
            noc_async_write(header_addr, relay_buffer_noc, packet_header_size_bytes);
            noc_async_write(packet_addr, relay_buffer_noc + packet_header_size_bytes, total_payload_size);
            noc_async_write_barrier();

            // Signal relay master
            uint64_t relay_sem_noc = get_noc_addr(relay_master_noc_x, relay_master_noc_y, my_relay_sem_addr);
            noc_semaphore_inc(relay_sem_noc, 1);
            noc_async_atomic_barrier();
        }

        // ======================================================================
        // R2 PHASE: Wait for compute, then send R1 result to relay master
        // ======================================================================
        {
            DeviceZoneScopedN("RELAY-R2-WAIT-COMPUTE");
            // Wait for compute sync
            cb_wait_front(cb_sync, 1);
            cb_pop_front(cb_sync, 1);

            cb_wait_front(cb_r1_result_l, out_tiles);
            cb_wait_front(cb_r1_result_s, Sq_chunk_t);
            cb_wait_front(cb_r1_result_m, Sq_chunk_t);
        }

        {
            DeviceZoneScopedN("RELAY-R2-PREPARE");

            // Prepare R2 packet header
            cb_reserve_back(packet_header_cb_id, 1);
            uint32_t header_addr = get_write_ptr(packet_header_cb_id);
            auto* header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_addr);
            fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)header, num_hops);

            // Build fused packet header for R2 destination
            const uint64_t r2_dst_noc = get_noc_addr(current_core_x, current_core_y, r2_neighbor_dst_addr);
            const uint64_t r2_sem_noc = get_noc_addr(current_core_x, current_core_y, r2_neighbor_sem_addr);
            header->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{r2_dst_noc, r2_sem_noc, 1, false},
                align(total_payload_size, l1_alignment));

            cb_push_back(packet_header_cb_id, 1);

            // Pack R1 result into payload
            cb_reserve_back(packet_cb_id, 1);
            uint32_t packet_addr = get_write_ptr(packet_cb_id);
            tt_memmove<true, false, false, 0>(packet_addr, get_read_ptr(cb_r1_result_l), payload_size_bytes);
            tt_memmove<true, false, false, 0>(
                packet_addr + payload_size_bytes, get_read_ptr(cb_r1_result_s), aligned_page_size);
            tt_memmove<true, false, false, 0>(
                packet_addr + payload_size_bytes + aligned_page_size, get_read_ptr(cb_r1_result_m), aligned_page_size);
            cb_push_back(packet_cb_id, 1);

            // NOC write to relay master's relay buffer
            uint64_t relay_buffer_noc = get_noc_addr(relay_master_noc_x, relay_master_noc_y, my_relay_buffer_addr);
            noc_async_write(header_addr, relay_buffer_noc, packet_header_size_bytes);
            noc_async_write(packet_addr, relay_buffer_noc + packet_header_size_bytes, total_payload_size);
            noc_async_write_barrier();

            // Signal relay master
            uint64_t relay_sem_noc = get_noc_addr(relay_master_noc_x, relay_master_noc_y, my_relay_sem_addr);
            noc_semaphore_inc(relay_sem_noc, 1);
            noc_async_atomic_barrier();
        }

        // Relay workers are done
        return;
    }

    // ==========================================================================
    // RELAY MASTER PATH: Send own packets + relay packets from workers
    // ==========================================================================

    // Initialize relay semaphore pointers
    volatile tt_l1_ptr uint32_t* relay_sem_ptrs[3];
    for (uint32_t i = 0; i < num_relay_workers; i++) {
        relay_sem_ptrs[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(relay_sem_addrs[i]);
    }

    // Track which relay workers have sent packets (for polling)
    bool relay_r1_done[3] = {false, false, false};
    uint32_t relay_r1_count = 0;

    // ==========================================================================
    // R1 PHASE: Send own R1 packet, then poll and forward relay R1 packets
    // ==========================================================================

    // Build direct fabric router connection
    // The runtime args were appended by append_fabric_connection_rt_args in program factory
    auto r1_router = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

    {
        DeviceZoneScopedN("MASTER-R1-ROUTER-CONNECT");
        r1_router.open();
    }

    // Prepare and send own R1 packet
    {
        DeviceZoneScopedN("MASTER-R1-SEND-OWN");

        cb_reserve_back(packet_header_cb_id, 1);
        uint32_t header_addr = get_write_ptr(packet_header_cb_id);
        auto* header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_addr);
        fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)header, num_hops);

        cb_reserve_back(packet_cb_id, 1);
        uint32_t packet_addr = get_write_ptr(packet_cb_id);
        tt_memmove<true, false, false, 0>(packet_addr, src_addr_l, payload_size_bytes);
        tt_memmove<true, false, false, 0>(packet_addr + payload_size_bytes, src_addr_s, aligned_page_size);
        tt_memmove<true, false, false, 0>(
            packet_addr + payload_size_bytes + aligned_page_size, src_addr_m, aligned_page_size);

        const uint64_t r1_dst_noc = get_noc_addr(current_core_x, current_core_y, r1_neighbor_dst_addr);
        const uint64_t r1_sem_noc = get_noc_addr(current_core_x, current_core_y, r1_neighbor_sem_addr);
        header->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{r1_dst_noc, r1_sem_noc, 1, false},
            align(total_payload_size, l1_alignment));

        r1_router.wait_for_empty_write_slot();
        r1_router.send_payload_without_header_non_blocking_from_address(packet_addr, total_payload_size);
        r1_router.send_payload_flush_non_blocking_from_address(header_addr, packet_header_size_bytes);

        cb_push_back(packet_header_cb_id, 1);
        cb_push_back(packet_cb_id, 1);
    }

    // Poll and forward relay R1 packets
    {
        DeviceZoneScopedN("MASTER-R1-RELAY");

        while (relay_r1_count < num_relay_workers) {
            for (uint32_t i = 0; i < num_relay_workers; i++) {
                if (relay_r1_done[i]) {
                    continue;
                }

                // Check if relay worker i has sent its packet
                if (*relay_sem_ptrs[i] > 0) {
                    noc_semaphore_set(relay_sem_ptrs[i], 0);

                    // Forward relay packet to fabric router
                    // Relay buffer format: [header][payload]
                    uint32_t relay_header_addr = relay_buffer_addrs[i];
                    uint32_t relay_payload_addr = relay_buffer_addrs[i] + packet_header_size_bytes;

                    r1_router.wait_for_empty_write_slot();
                    r1_router.send_payload_without_header_non_blocking_from_address(
                        relay_payload_addr, total_payload_size);
                    r1_router.send_payload_flush_blocking_from_address(relay_header_addr, packet_header_size_bytes);

                    relay_r1_done[i] = true;
                    relay_r1_count++;
                }
            }
        }
    }

    // ==========================================================================
    // R2 PHASE SETUP: Connect to R2 fabric router while waiting for compute
    // ==========================================================================

    // Build direct fabric router connection for R2
    auto r2_router = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);

    {
        DeviceZoneScopedN("MASTER-R2-ROUTER-CONNECT");
        r2_router.open();
    }

    // ==========================================================================
    // R2 PHASE: Wait for compute, send own R2, poll and forward relay R2 packets
    // ==========================================================================

    // Reset relay tracking for R2
    bool relay_r2_done[3] = {false, false, false};
    uint32_t relay_r2_count = 0;

    // Wait for compute sync
    {
        DeviceZoneScopedN("MASTER-R2-WAIT-COMPUTE");
        cb_wait_front(cb_sync, 1);
        cb_pop_front(cb_sync, 1);

        cb_wait_front(cb_r1_result_l, out_tiles);
        cb_wait_front(cb_r1_result_s, Sq_chunk_t);
        cb_wait_front(cb_r1_result_m, Sq_chunk_t);
    }

    // Send own R2 packet
    {
        DeviceZoneScopedN("MASTER-R2-SEND-OWN");

        cb_reserve_back(packet_header_cb_id, 1);
        uint32_t header_addr = get_write_ptr(packet_header_cb_id);
        auto* header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_addr);
        fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)header, num_hops);

        cb_reserve_back(packet_cb_id, 1);
        uint32_t packet_addr = get_write_ptr(packet_cb_id);
        tt_memmove<true, false, false, 0>(packet_addr, get_read_ptr(cb_r1_result_l), payload_size_bytes);
        tt_memmove<true, false, false, 0>(
            packet_addr + payload_size_bytes, get_read_ptr(cb_r1_result_s), aligned_page_size);
        tt_memmove<true, false, false, 0>(
            packet_addr + payload_size_bytes + aligned_page_size, get_read_ptr(cb_r1_result_m), aligned_page_size);

        const uint64_t r2_dst_noc = get_noc_addr(current_core_x, current_core_y, r2_neighbor_dst_addr);
        const uint64_t r2_sem_noc = get_noc_addr(current_core_x, current_core_y, r2_neighbor_sem_addr);
        header->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{r2_dst_noc, r2_sem_noc, 1, false},
            align(total_payload_size, l1_alignment));

        r2_router.wait_for_empty_write_slot();
        r2_router.send_payload_without_header_non_blocking_from_address(packet_addr, total_payload_size);
        r2_router.send_payload_flush_non_blocking_from_address(header_addr, packet_header_size_bytes);

        cb_push_back(packet_header_cb_id, 1);
        cb_push_back(packet_cb_id, 1);
    }

    // Poll and forward relay R2 packets
    {
        DeviceZoneScopedN("MASTER-R2-RELAY");

        while (relay_r2_count < num_relay_workers) {
            for (uint32_t i = 0; i < num_relay_workers; i++) {
                if (relay_r2_done[i]) {
                    continue;
                }

                // Check if relay worker i has sent its R2 packet
                if (*relay_sem_ptrs[i] > 0) {
                    noc_semaphore_set(relay_sem_ptrs[i], 0);

                    // Forward relay packet to fabric router
                    uint32_t relay_header_addr = relay_buffer_addrs[i];
                    uint32_t relay_payload_addr = relay_buffer_addrs[i] + packet_header_size_bytes;

                    r2_router.wait_for_empty_write_slot();
                    r2_router.send_payload_without_header_non_blocking_from_address(
                        relay_payload_addr, total_payload_size);
                    r2_router.send_payload_flush_blocking_from_address(relay_header_addr, packet_header_size_bytes);

                    relay_r2_done[i] = true;
                    relay_r2_count++;
                }
            }
        }
    }

    // ==========================================================================
    // DISCONNECT FROM FABRIC ROUTERS
    // ==========================================================================
    // Close the direct router connections
    {
        DeviceZoneScopedN("MASTER-ROUTER-DISCONNECT");
        r1_router.close();
        r2_router.close();
    }

    noc_async_write_barrier();
}
