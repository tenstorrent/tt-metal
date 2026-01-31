// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
// Writer kernel for reduce_to_all operation using AGGREGATOR (combined MS format).
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

// =============================================================================
// Helper functions for packet construction and forwarding
// =============================================================================

/**
 * Prepare fabric packet header with routing and fused write+atomic_inc command.
 * No data dependency - can be called before payload data is ready.
 */
template <uint32_t l1_alignment, uint32_t total_payload_size>
FORCE_INLINE void prepare_header(
    uint32_t header_addr, uint32_t dst_mesh_id, uint32_t dst_chip_id, uint64_t dst_noc, uint64_t sem_noc) {
    auto* header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_addr);
    constexpr uint32_t ATOMIC_INC_VAL = 1;
    constexpr bool FLUSH_WRITES = false;

    DPRINT << "Preparing packet header for dst_mesh_id=" << dst_mesh_id << ", dst_chip_id=" << dst_chip_id
           << ", dst_noc=" << dst_noc << ", sem_noc=" << sem_noc << ENDL();

    (void)fabric_set_unicast_route(header, dst_chip_id, dst_mesh_id);
    header->to_noc_fused_unicast_write_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc, sem_noc, ATOMIC_INC_VAL, FLUSH_WRITES},
        align(total_payload_size, l1_alignment));
}

/**
 * Pack L and MS data into contiguous packet buffer after header.
 * Layout: [HEADER][L tiles][MS tile aligned]
 */
template <uint32_t payload_size_bytes, uint32_t ms_tile_size_bytes>
FORCE_INLINE void pack_payload(uint32_t slot_addr, uint32_t src_l, uint32_t src_ms) {
    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
    uint32_t payload_start = slot_addr + packet_header_size_bytes;

    DPRINT << "payload size_bytes=" << payload_size_bytes << ", ms_tile_size_bytes=" << ms_tile_size_bytes << ENDL();

    tt_memmove<true, false, false, 0>(payload_start, src_l, payload_size_bytes);
    tt_memmove<true, false, false, 0>(payload_start + payload_size_bytes, src_ms, ms_tile_size_bytes);
}

/**
 * Forward complete packet (header + payload) to aggregator slot in single NOC transfer.
 * Uses bit-packed semaphore signaling: increments by (1 << slot_idx) so aggregator
 * can identify exactly which slot is ready.
 */
template <uint32_t slot_size>
FORCE_INLINE void forward_packet(uint32_t slot_addr, uint64_t agg_slot_noc, uint64_t agg_sem_noc, uint32_t slot_idx) {
    DPRINT << "forward_packet: slot_addr=" << slot_addr << ", slot_idx=" << slot_idx << ", slot_size=" << slot_size
           << ENDL();
    noc_async_write(slot_addr, agg_slot_noc, slot_size);
    noc_async_writes_flushed();
    noc_semaphore_inc(agg_sem_noc, 1u << slot_idx);
    DPRINT << "forward_packet: signaled aggregator with bit " << (1u << slot_idx) << ENDL();
}

// =============================================================================
// Main kernel
// =============================================================================

void kernel_main() {
    // ==========================================================================
    // Compile-time args (combined MS format)
    // ==========================================================================
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t vDHt = get_compile_time_arg_val(1);

    // CB IDs
    [[maybe_unused]] constexpr uint32_t cb_local_l = get_compile_time_arg_val(2);   // UNUSED - read from tensor
    [[maybe_unused]] constexpr uint32_t cb_local_ms = get_compile_time_arg_val(3);  // UNUSED
    constexpr uint32_t cb_r1_result_l = get_compile_time_arg_val(4);                // R1 compute output for R2 send
    constexpr uint32_t cb_r1_result_ms = get_compile_time_arg_val(5);

    // Unified packet slot CB (header + payload in single buffer)
    constexpr uint32_t cb_packet_slot = get_compile_time_arg_val(6);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(7);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(9);
    constexpr uint32_t slot_size = get_compile_time_arg_val(10);            // L1-aligned slot size
    constexpr uint32_t cb_sync_writer_done = get_compile_time_arg_val(11);  // Writer -> Compute sync

    // Derived constants
    constexpr uint32_t out_tiles = Sq_chunk_t * vDHt;
    constexpr uint32_t payload_size_bytes = out_tiles * page_size_bytes;  // L payload
    constexpr uint32_t aligned_page_size = ((page_size_bytes + l1_alignment - 1) / l1_alignment) * l1_alignment;
    constexpr uint32_t ms_tile_size_bytes = aligned_page_size;  // Single combined MS tile
    constexpr uint32_t total_payload_size = payload_size_bytes + ms_tile_size_bytes;

    // ==========================================================================
    // Runtime args (combined MS format)
    // ==========================================================================
    size_t arg_idx = 0;

    // Input tensor source addresses (for R1 send - read directly, no CB sync)
    const uint32_t src_addr_l = get_arg_val<uint32_t>(arg_idx++);   // Local L source
    const uint32_t src_addr_ms = get_arg_val<uint32_t>(arg_idx++);  // Local MS source

    // R1 destination - intermediate tensor address on neighbor device
    const uint32_t r1_dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_dst_chip_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_neighbor_dst_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);

    // R2 destination - intermediate tensor address on neighbor device
    const uint32_t r2_dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_dst_chip_id = get_arg_val<uint32_t>(arg_idx++);
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
    const uint32_t r1_slot_idx = get_arg_val<uint32_t>(arg_idx++);  // For bit-packed signaling
    const uint32_t r2_agg_slot_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_agg_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t r2_slot_idx = get_arg_val<uint32_t>(arg_idx++);  // For bit-packed signaling

    // ==========================================================================
    // Precompute NOC addresses (avoids redundant calculations in helpers)
    // ==========================================================================
    const uint64_t r1_dst_noc = get_noc_addr(current_core_x, current_core_y, r1_neighbor_dst_addr);
    const uint64_t r1_sem_noc = get_noc_addr(current_core_x, current_core_y, r1_neighbor_sem_addr);
    const uint64_t r1_agg_slot_noc = get_noc_addr(agg_core_x, agg_core_y, r1_agg_slot_addr);
    const uint64_t r1_agg_sem_noc = get_noc_addr(agg_core_x, agg_core_y, r1_agg_sem_addr);

    const uint64_t r2_dst_noc = get_noc_addr(current_core_x, current_core_y, r2_neighbor_dst_addr);
    const uint64_t r2_sem_noc = get_noc_addr(current_core_x, current_core_y, r2_neighbor_sem_addr);
    const uint64_t r2_agg_slot_noc = get_noc_addr(agg_core_x, agg_core_y, r2_agg_slot_addr);
    const uint64_t r2_agg_sem_noc = get_noc_addr(agg_core_x, agg_core_y, r2_agg_sem_addr);

    DPRINT << "Writer starting R1 send to mesh " << r1_dst_mesh_id << " chip " << r1_dst_chip_id << ENDL();

    // ==========================================================================
    // ROUND 1: Send local input to R1 neighbor via aggregator
    // ==========================================================================
    {
        DeviceZoneScopedN("R1-SEND");

        cb_reserve_back(cb_packet_slot, 1);
        uint32_t slot_addr = get_write_ptr(cb_packet_slot);

        prepare_header<l1_alignment, total_payload_size>(
            slot_addr, r1_dst_mesh_id, r1_dst_chip_id, r1_dst_noc, r1_sem_noc);
        pack_payload<payload_size_bytes, ms_tile_size_bytes>(slot_addr, src_addr_l, src_addr_ms);
        forward_packet<slot_size>(slot_addr, r1_agg_slot_noc, r1_agg_sem_noc, r1_slot_idx);

        cb_push_back(cb_packet_slot, 1);
    }

    DPRINT << "Writer R1 send complete, preparing R2 send after compute." << ENDL();

    // ==========================================================================
    // R2 Header Preparation (overlapped with compute - no data dependency)
    // ==========================================================================
    uint32_t r2_slot_addr;
    {
        DeviceZoneScopedN("R2-PREPARE-HEADER");

        cb_reserve_back(cb_packet_slot, 1);
        r2_slot_addr = get_write_ptr(cb_packet_slot);

        prepare_header<l1_alignment, total_payload_size>(
            r2_slot_addr, r2_dst_mesh_id, r2_dst_chip_id, r2_dst_noc, r2_sem_noc);
    }

    // ==========================================================================
    // Wait for R1 compute result
    // ==========================================================================
    {
        DeviceZoneScopedN("R2-WAIT-COMPUTE");

        DPRINT << "Writer waiting for cb_sync" << ENDL();
        cb_wait_front(cb_sync, 1);
        DPRINT << "Writer cb_sync acquired" << ENDL();
        cb_pop_front(cb_sync, 1);

        DPRINT << "Writer waiting for cb_r1_result_l (" << out_tiles << " tiles)" << ENDL();
        cb_wait_front(cb_r1_result_l, out_tiles);
        DPRINT << "Writer waiting for cb_r1_result_ms" << ENDL();
        cb_wait_front(cb_r1_result_ms, 1);
        DPRINT << "Writer got R1 results from compute" << ENDL();
    }

    // ==========================================================================
    // ROUND 2: Send R1 result to R2 neighbor via aggregator
    // ==========================================================================
    {
        DeviceZoneScopedN("R2-PACK-SEND");

        // Pack R1 results into packet buffer (memcpy from CB to local buffer)
        pack_payload<payload_size_bytes, ms_tile_size_bytes>(
            r2_slot_addr, get_read_ptr(cb_r1_result_l), get_read_ptr(cb_r1_result_ms));

        // Signal compute that we're done reading R1 results - it can now pop them
        cb_reserve_back(cb_sync_writer_done, 1);
        cb_push_back(cb_sync_writer_done, 1);

        forward_packet<slot_size>(r2_slot_addr, r2_agg_slot_noc, r2_agg_sem_noc, r2_slot_idx);

        cb_push_back(cb_packet_slot, 1);
    }

    noc_async_full_barrier();
}
