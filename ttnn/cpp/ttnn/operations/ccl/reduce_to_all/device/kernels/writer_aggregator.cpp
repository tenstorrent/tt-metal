// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
// Writer kernel for reduce_to_all operation using AGGREGATOR.
// This kernel runs on SHARD CORES (data cores).
//
// AGGREGATOR APPROACH:
// Instead of using heavyweight mux with connect/disconnect cycles, workers write
// complete packets to aggregator slots via NoC and increment aggregator semaphore.
// The aggregator kernel (running on aggregator cores) collects packets and forwards
// them via fabric. This dramatically reduces per-worker overhead.
//
// ZERO-COPY OPTIMIZATION:
// Destination CBs on neighbor are backed by MeshBuffer (same L1 address on all devices).
// Packets include fabric header with fused write+sem_inc for direct delivery.
//
// TYPE A/B WORKER SPLIT:
// Workers are classified as Type A or Type B based on (device_id + core_index) % 2:
//   - Type A: R1 sends via FWD aggregator, R2 sends via BWD aggregator
//   - Type B: R1 sends via BWD aggregator, R2 sends via FWD aggregator
// This balances FWD/BWD traffic in each round.

#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/debug/dprint.h"
#include <cstdint>

using tt::data_movement::common::tt_memmove;

void kernel_main() {
    // ==========================================================================
    // Compile-time args
    // ==========================================================================
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t vDHt = get_compile_time_arg_val(1);

    // CB IDs (kept for compatibility - some are unused in writer)
    [[maybe_unused]] constexpr uint32_t cb_local_l = get_compile_time_arg_val(2);  // UNUSED - read from tensor
    [[maybe_unused]] constexpr uint32_t cb_local_s = get_compile_time_arg_val(3);  // UNUSED
    [[maybe_unused]] constexpr uint32_t cb_local_m = get_compile_time_arg_val(4);  // UNUSED
    constexpr uint32_t cb_r1_result_l = get_compile_time_arg_val(5);               // R1 compute output for R2 send
    constexpr uint32_t cb_r1_result_s = get_compile_time_arg_val(6);
    constexpr uint32_t cb_r1_result_m = get_compile_time_arg_val(7);

    // Packet/header CBs
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(9);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(10);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(12);
    constexpr uint32_t slot_size = get_compile_time_arg_val(13);  // Aggregator slot size

    // Derived constants
    constexpr uint32_t out_tiles = Sq_chunk_t * vDHt;
    constexpr uint32_t payload_size_bytes = out_tiles * page_size_bytes;  // L payload
    constexpr uint32_t aligned_page_size = ((page_size_bytes + l1_alignment - 1) / l1_alignment) * l1_alignment;
    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
    constexpr uint8_t num_hops = 1;
    constexpr uint32_t total_payload_size = payload_size_bytes + 2 * aligned_page_size;

    // ==========================================================================
    // Runtime args
    // ==========================================================================
    size_t arg_idx = 0;

    // Input tensor source addresses (for R1 send - read directly, no CB sync)
    const uint32_t src_addr_l = get_arg_val<uint32_t>(arg_idx++);  // Local L source
    const uint32_t src_addr_s = get_arg_val<uint32_t>(arg_idx++);  // Local S source
    const uint32_t src_addr_m = get_arg_val<uint32_t>(arg_idx++);  // Local M source

    // R1 destination - intermediate tensor address on neighbor device
    const uint32_t r1_neighbor_dst_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);

    // R2 destination - intermediate tensor address on neighbor device
    const uint32_t r2_neighbor_dst_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);

    // Local core coordinates (for NOC address calculation in fused packets)
    const uint32_t current_core_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t current_core_y = get_arg_val<uint32_t>(arg_idx++);

    // Aggregator core and slot info
    const uint32_t agg_core_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t agg_core_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_agg_slot_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_agg_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t r2_agg_slot_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_agg_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    DPRINT << "Writer: agg_core=(" << agg_core_x << "," << agg_core_y << ")" << ENDL();
    DPRINT << "Writer: r1_slot=" << r1_agg_slot_addr << " r1_sem=" << r1_agg_sem_addr << ENDL();
    DPRINT << "Writer: r2_slot=" << r2_agg_slot_addr << " r2_sem=" << r2_agg_sem_addr << ENDL();

    // ==========================================================================
    // PHASE 1: Send R1 (local input to R1 neighbor via aggregator)
    // ==========================================================================
    DPRINT << "Writer: Starting R1 phase..." << ENDL();
    {
        DeviceZoneScopedN("R1-PACK-DATA");

        // Reserve space for packet in local CB
        cb_reserve_back(packet_cb_id, 1);
        uint32_t packet_addr = get_write_ptr(packet_cb_id);

        // Build packet header (fused write + sem_inc to neighbor)
        cb_reserve_back(packet_header_cb_id, 1);
        uint32_t header_addr = get_write_ptr(packet_header_cb_id);
        auto* header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_addr);

        // Set routing for fabric unicast
        fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)header, num_hops);

        // Fused command: write data to neighbor CB + increment neighbor semaphore
        const uint64_t r1_neighbor_dst_noc = get_noc_addr(current_core_x, current_core_y, r1_neighbor_dst_addr);
        const uint64_t r1_neighbor_sem_noc = get_noc_addr(current_core_x, current_core_y, r1_neighbor_sem_addr);
        header->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{r1_neighbor_dst_noc, r1_neighbor_sem_noc, 1, false},
            align(total_payload_size, l1_alignment));

        // Pack payload: [L tiles][S tile aligned][M tile aligned]
        // Read directly from input tensor addresses (no CB sync to avoid contention with compute)
        tt_memmove<true, false, false, 0>(packet_addr, src_addr_l, payload_size_bytes);
        tt_memmove<true, false, false, 0>(packet_addr + payload_size_bytes, src_addr_s, aligned_page_size);
        tt_memmove<true, false, false, 0>(
            packet_addr + payload_size_bytes + aligned_page_size, src_addr_m, aligned_page_size);

        cb_push_back(packet_header_cb_id, 1);
        cb_push_back(packet_cb_id, 1);

        // Write complete packet to aggregator slot via NoC
        // Aggregator expects: [header][payload] contiguous in slot
        uint64_t agg_slot_noc = get_noc_addr(agg_core_x, agg_core_y, r1_agg_slot_addr);

        // Write header
        noc_async_write(header_addr, agg_slot_noc, packet_header_size_bytes);
        // Write payload after header
        noc_async_write(packet_addr, agg_slot_noc + packet_header_size_bytes, total_payload_size);

        noc_async_writes_flushed();  // Ensure write completes before signaling

        // Increment aggregator semaphore to signal packet ready
        uint64_t agg_sem_noc = get_noc_addr(agg_core_x, agg_core_y, r1_agg_sem_addr);
        noc_semaphore_inc(agg_sem_noc, 1);
        DPRINT << "Writer: R1 packet sent, sem incremented" << ENDL();
    }

    // ==========================================================================
    // PHASE 2: Wait for R1 compute result
    // ==========================================================================
    DPRINT << "Writer: Waiting for compute sync..." << ENDL();
    {
        DeviceZoneScopedN("R2-WAIT-COMPUTE");

        // Wait for SYNC signal from Compute
        cb_wait_front(cb_sync, 1);
        cb_pop_front(cb_sync, 1);
        DPRINT << "Writer: Compute sync received, waiting for R1 result CBs..." << ENDL();

        // Wait for R1 result data to be ready
        cb_wait_front(cb_r1_result_l, out_tiles);
        cb_wait_front(cb_r1_result_s, Sq_chunk_t);
        cb_wait_front(cb_r1_result_m, Sq_chunk_t);
        DPRINT << "Writer: R1 result CBs ready" << ENDL();
    }

    // ==========================================================================
    // PHASE 3: Send R2 (R1 result to R2 neighbor via aggregator)
    // ==========================================================================
    {
        DeviceZoneScopedN("R2-PACK-DATA");

        // Reserve space for packet
        cb_reserve_back(packet_cb_id, 1);
        uint32_t packet_addr = get_write_ptr(packet_cb_id);

        // Build packet header
        cb_reserve_back(packet_header_cb_id, 1);
        uint32_t header_addr = get_write_ptr(packet_header_cb_id);
        auto* header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_addr);

        fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)header, num_hops);

        const uint64_t r2_neighbor_dst_noc = get_noc_addr(current_core_x, current_core_y, r2_neighbor_dst_addr);
        const uint64_t r2_neighbor_sem_noc = get_noc_addr(current_core_x, current_core_y, r2_neighbor_sem_addr);
        header->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{r2_neighbor_dst_noc, r2_neighbor_sem_noc, 1, false},
            align(total_payload_size, l1_alignment));

        // Pack R1 result into payload
        tt_memmove<true, false, false, 0>(packet_addr, get_read_ptr(cb_r1_result_l), payload_size_bytes);
        tt_memmove<true, false, false, 0>(
            packet_addr + payload_size_bytes, get_read_ptr(cb_r1_result_s), aligned_page_size);
        tt_memmove<true, false, false, 0>(
            packet_addr + payload_size_bytes + aligned_page_size, get_read_ptr(cb_r1_result_m), aligned_page_size);

        cb_push_back(packet_header_cb_id, 1);
        cb_push_back(packet_cb_id, 1);

        // Write packet to R2 aggregator slot
        uint64_t agg_slot_noc = get_noc_addr(agg_core_x, agg_core_y, r2_agg_slot_addr);

        noc_async_write(header_addr, agg_slot_noc, packet_header_size_bytes);
        noc_async_write(packet_addr, agg_slot_noc + packet_header_size_bytes, total_payload_size);

        noc_async_writes_flushed();

        // Signal R2 packet ready to aggregator
        uint64_t agg_sem_noc = get_noc_addr(agg_core_x, agg_core_y, r2_agg_sem_addr);
        noc_semaphore_inc(agg_sem_noc, 1);
        DPRINT << "Writer: R2 packet sent, sem incremented" << ENDL();
    }

    DPRINT << "Writer: Done!" << ENDL();
    // NOTE: No mux termination needed - aggregator handles fabric connection lifecycle.
    // This is a key benefit of the aggregator approach: workers don't need to manage
    // per-client fabric connection setup/teardown.
}
