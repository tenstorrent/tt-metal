// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
// Simplified writer kernel for reduce_to_all operation with INTERLEAVED mux ops.
// This kernel runs on SHARD CORES (data cores).
//
// ZERO-COPY OPTIMIZATION:
// Destination CBs on neighbor are backed by MeshBuffer (same L1 address on all devices).
// Writer sends data DIRECTLY to neighbor's CB - no intermediate buffer, no memcpy on receiver!
//
// R1/R2 MUX DIRECTION:
// The program factory determines which physical mux (FWD or BWD) to use for R1 and R2
// based on device position in the ring:
//   - Even devices (0,2): R1=FWD mux, R2=BWD mux
//   - Odd devices (1,3):  R1=BWD mux, R2=FWD mux
// This kernel receives R1 and R2 mux configs - it doesn't need to know about FWD/BWD!
//
// WHY SEPARATE R1 AND R2 DESTINATIONS:
//   R1 and R2 send to DIFFERENT neighbors:
//     - R1: pairs (D0,D1) and (D2,D3) exchange
//     - R2: pairs (D1,D2) and (D3,D0) exchange
//   Each neighbor has its own MeshBuffer for receiving.
//
// Key optimization: Interleave mux setup with compute to hide latency:
//
// Timeline:
//   R1 mux: [setup][wait_ready][connect][send R1 → neighbor CB]
//                                           │
//                                           ▼ waiting for compute...
//   R2 mux:                      [setup][wait_ready][connect]  ← hidden by compute!
//                                                       │
//   COMPUTE:                 ........[R1 processing]    │
//                                        │              │
//                                        ▼              ▼
//                                   R1 result  [send R2 → neighbor CB] ← immediate!
//
// Benefits:
// - R2 channel not held during R1 phase
// - R2 mux setup hidden by R1 compute time
// - Both terminations at end, can overlap with final output writes
// - Neighbor reader does NO memcpy - data arrives directly in their CB!

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
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
    // NOTE: cb_local_l/s/m are kept for compatibility but NOT USED!
    // The writer reads directly from input tensor addresses (passed as runtime args)
    // to avoid CB contention with the compute kernel which uses CB sync.
    [[maybe_unused]] constexpr uint32_t cb_local_l = get_compile_time_arg_val(2);  // UNUSED
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

    // Mux configuration indices - placed after base args so they can be auto-calculated
    // in the program factory based on vector size (robust to adding/removing base args)
    constexpr uint32_t r1_mux_ct_idx = get_compile_time_arg_val(13);
    constexpr uint32_t r2_mux_ct_idx = get_compile_time_arg_val(14);

    // Derived constants
    constexpr uint32_t out_tiles = Sq_chunk_t * vDHt;
    constexpr uint32_t payload_size_bytes = out_tiles * page_size_bytes;  // L payload

    // R1 mux compile-time config (actual FWD/BWD direction set by program factory)
    constexpr uint8_t r1_mux_num_buffers = get_compile_time_arg_val(r1_mux_ct_idx);
    constexpr size_t r1_mux_buffer_size = get_compile_time_arg_val(r1_mux_ct_idx + 1);
    constexpr size_t r1_mux_status_addr = get_compile_time_arg_val(r1_mux_ct_idx + 2);
    constexpr size_t r1_mux_term_addr = get_compile_time_arg_val(r1_mux_ct_idx + 3);
    constexpr uint32_t r1_num_clients = get_compile_time_arg_val(r1_mux_ct_idx + 4);

    // R2 mux compile-time config
    constexpr uint8_t r2_mux_num_buffers = get_compile_time_arg_val(r2_mux_ct_idx);
    constexpr size_t r2_mux_buffer_size = get_compile_time_arg_val(r2_mux_ct_idx + 1);
    constexpr size_t r2_mux_status_addr = get_compile_time_arg_val(r2_mux_ct_idx + 2);
    constexpr size_t r2_mux_term_addr = get_compile_time_arg_val(r2_mux_ct_idx + 3);
    constexpr uint32_t r2_num_clients = get_compile_time_arg_val(r2_mux_ct_idx + 4);

    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
    constexpr uint8_t num_hops = 1;
    constexpr uint32_t aligned_page_size = ((page_size_bytes + l1_alignment - 1) / l1_alignment) * l1_alignment;

    // ==========================================================================
    // Runtime args
    // ==========================================================================
    size_t arg_idx = 0;

    // Input tensor source addresses (for R1 send - read directly, no CB sync)
    // This avoids CB contention with compute kernel which uses CB sync
    const uint32_t src_addr_l = get_arg_val<uint32_t>(arg_idx++);  // Local L source
    const uint32_t src_addr_s = get_arg_val<uint32_t>(arg_idx++);  // Local S source
    const uint32_t src_addr_m = get_arg_val<uint32_t>(arg_idx++);  // Local M source

    // R1 destination - intermediate tensor address on neighbor device
    const uint32_t r1_neighbor_dst_addr = get_arg_val<uint32_t>(arg_idx++);  // R1 dest addr
    const uint32_t r1_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);  // R1 semaphore addr

    // R2 destination - intermediate tensor address on neighbor device (DIFFERENT from R1!)
    const uint32_t r2_neighbor_dst_addr = get_arg_val<uint32_t>(arg_idx++);  // R2 dest addr
    const uint32_t r2_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);  // R2 semaphore addr

    // Local core coordinates (for NOC address calculation in fused packets)
    const uint32_t current_core_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t current_core_y = get_arg_val<uint32_t>(arg_idx++);

    // R1 mux runtime args (physical direction determined by program factory)
    const bool r1_is_term_master = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t r1_mux_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t r1_mux_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t r1_mux_channel_base = get_arg_val<uint32_t>(arg_idx++);
    const size_t r1_mux_conn_info = get_arg_val<uint32_t>(arg_idx++);
    const size_t r1_mux_handshake = get_arg_val<uint32_t>(arg_idx++);
    const size_t r1_mux_flow_ctrl = get_arg_val<uint32_t>(arg_idx++);
    const size_t r1_mux_buf_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t r1_mux_channel_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t r1_term_sync_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t r1_local_status_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t r1_local_flow_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t r1_local_teardown_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t r1_local_buf_idx_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t r1_term_master_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t r1_term_master_y = get_arg_val<uint32_t>(arg_idx++);

    // R2 mux runtime args
    const bool r2_is_term_master = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t r2_mux_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t r2_mux_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t r2_mux_channel_base = get_arg_val<uint32_t>(arg_idx++);
    const size_t r2_mux_conn_info = get_arg_val<uint32_t>(arg_idx++);
    const size_t r2_mux_handshake = get_arg_val<uint32_t>(arg_idx++);
    const size_t r2_mux_flow_ctrl = get_arg_val<uint32_t>(arg_idx++);
    const size_t r2_mux_buf_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t r2_mux_channel_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t r2_term_sync_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t r2_local_status_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t r2_local_flow_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t r2_local_teardown_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t r2_local_buf_idx_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t r2_term_master_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t r2_term_master_y = get_arg_val<uint32_t>(arg_idx++);

    // NOTE: Barrier leader config removed - not needed in simplified design.
    // Mux termination provides sufficient synchronization.

    // Computed values (S and M tiles are single tiles, aligned)
    const uint32_t total_payload_size = payload_size_bytes + 2 * aligned_page_size;

    uint32_t r1_header_addr = get_write_ptr(packet_header_cb_id);
    auto* r1_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(r1_header_addr);

    // ==========================================================================
    // PHASE 1: Setup R1 mux, connect, and send R1
    // ==========================================================================
    {
        DeviceZoneScopedN("R1-MUX-SETUP");
        // Setup R1 packet header
        cb_reserve_back(packet_header_cb_id, 1);
        cb_push_back(packet_header_cb_id, 1);
        fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)r1_header, num_hops);
    }

    // Build R1 mux connection
    auto r1_mux = tt::tt_fabric::build_connection_to_fabric_endpoint<r1_mux_num_buffers>(
        r1_mux_x,
        r1_mux_y,
        r1_mux_channel_id,
        r1_mux_num_buffers,
        r1_mux_buffer_size,
        r1_mux_channel_base,
        r1_mux_conn_info,
        r1_mux_handshake,
        r1_mux_flow_ctrl,
        r1_mux_buf_idx,
        r1_local_flow_addr,
        r1_local_teardown_addr,
        r1_local_buf_idx_addr);

    // R1 SEND: Pack and send local input data to R1 neighbor
    //
    // IMPORTANT: We read directly from input tensor addresses (src_addr_l/s/m)
    // instead of using CB sync (cb_wait_front/cb_pop_front). This is critical!
    //
    // The compute kernel uses CB sync on the same CBs (cb_local_l/s/m).
    // If we also used CB sync here, we'd have a race condition - both writer
    // and compute would try to consume the same CB data.
    //
    // Since the input CBs are aliased to the input tensor, reading directly
    // from the tensor addresses gives us the same data without CB contention.

    cb_reserve_back(packet_cb_id, 1);
    uint32_t packet_addr = get_write_ptr(packet_cb_id);

    {
        DeviceZoneScopedN("R1-PACK-DATA");
        DPRINT << "writer packing data for R1" << ENDL();
        // Pack data: [L tiles][S tile aligned][M tile aligned]
        // Read directly from input tensor shard addresses (no CB sync!)
        tt_memmove<true, false, false, 0>(packet_addr, src_addr_l, payload_size_bytes);
        tt_memmove<true, false, false, 0>(packet_addr + payload_size_bytes, src_addr_s, aligned_page_size);
        tt_memmove<true, false, false, 0>(
            packet_addr + payload_size_bytes + aligned_page_size, src_addr_m, aligned_page_size);
    }

    // Build fused packet header for R1 - goes to R1 neighbor
    //
    // DESTINATION CORE ASSUMPTION:
    // We use current_core_x/y (our local core coords) as the destination core on the
    // remote device. This assumes SYMMETRIC SHARD PLACEMENT: core (x,y) on this device
    // sends to core (x,y) on the neighbor device. This works because:
    //   1. All devices have identical shard grid topology
    //   2. Intermediate tensor allocates the same L1 address on all devices
    //   3. Each shard core sends to its corresponding core on the neighbor
    //
    // The fabric routes the packet to the correct device; the NOC address specifies
    // where on that device the data should land.
    const uint64_t r1_neighbor_dst_noc = get_noc_addr(current_core_x, current_core_y, r1_neighbor_dst_addr);
    const uint64_t r1_neighbor_sem_noc = get_noc_addr(current_core_x, current_core_y, r1_neighbor_sem_addr);
    r1_header->to_noc_fused_unicast_write_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{r1_neighbor_dst_noc, r1_neighbor_sem_noc, 1, false},
        align(total_payload_size, l1_alignment));

    {
        DeviceZoneScopedN("R1-MUX-CONNECT");
        // Wait for R1 mux ready and connect
        DPRINT << "writer waiting for R1 mux ready" << ENDL();
        tt::tt_fabric::wait_for_fabric_endpoint_ready(r1_mux_x, r1_mux_y, r1_mux_status_addr, r1_local_status_addr);
    }

    {
        DeviceZoneScopedN("R1-MUX-SEND");
        // finish connect
        tt::tt_fabric::fabric_client_connect(r1_mux);

        // Send R1 via R1 mux
        // we can avoid waiting for an empty slot since we only have one packet to send and
        // atleast 1 buffer in the mux, and this is the first packet being sent.
        // r1_mux.wait_for_empty_write_slot();
        r1_mux.send_payload_without_header_non_blocking_from_address(packet_addr, total_payload_size);
        r1_mux.send_payload_flush_non_blocking_from_address((uint32_t)r1_header, packet_header_size_bytes);
    }

    cb_push_back(packet_cb_id, 1);  // Release packet buffer for reuse

    // ==========================================================================
    // PHASE 2: While waiting for R1 compute, setup R2 mux (hidden latency!)
    // ==========================================================================
    uint32_t r2_header_addr = get_write_ptr(packet_header_cb_id);
    auto* r2_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(r2_header_addr);

    {
        DeviceZoneScopedN("R2-MUX-SETUP");
        // Setup R2 packet header (compute is running in parallel on TRISC!)
        cb_reserve_back(packet_header_cb_id, 1);
        cb_push_back(packet_header_cb_id, 1);
        fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)r2_header, num_hops);
    }

    // Build R2 mux connection (compute still running!)
    auto r2_mux = tt::tt_fabric::build_connection_to_fabric_endpoint<r2_mux_num_buffers>(
        r2_mux_x,
        r2_mux_y,
        r2_mux_channel_id,
        r2_mux_num_buffers,
        r2_mux_buffer_size,
        r2_mux_channel_base,
        r2_mux_conn_info,
        r2_mux_handshake,
        r2_mux_flow_ctrl,
        r2_mux_buf_idx,
        r2_local_flow_addr,
        r2_local_teardown_addr,
        r2_local_buf_idx_addr);

    {
        DeviceZoneScopedN("R2-MUX-CONNECT");
        // Wait for R2 mux ready (should already be ready by now since compute took time!)
        tt::tt_fabric::wait_for_fabric_endpoint_ready(r2_mux_x, r2_mux_y, r2_mux_status_addr, r2_local_status_addr);
        tt::tt_fabric::fabric_client_connect(r2_mux);
    }

    // ==========================================================================
    // PHASE 3: Wait for R1 compute result, then send R2 (R2 mux already connected!)
    // ==========================================================================

    // Now wait for R1 compute to complete (R2 mux is already connected!)
    {
        DeviceZoneScopedN("R2-WAIT-COMPUTE");
        DPRINT << "writer waiting for sync signal from compute" << ENDL();
        // Wait for SYNC signal from Compute
        // This ensures R1 computation is FULLY DONE and cb_r1_result contains valid final data
        // even if the Read Pointer was exposed to partial data during accumulation.
        cb_wait_front(cb_sync, 1);
        cb_pop_front(cb_sync, 1);

        DPRINT << "writer got sync signal from compute" << ENDL();
        cb_wait_front(cb_r1_result_l, out_tiles);
        cb_wait_front(cb_r1_result_s, Sq_chunk_t);
        cb_wait_front(cb_r1_result_m, Sq_chunk_t);
    }

    // Prepare R2 packet with R1 result
    cb_reserve_back(packet_cb_id, 1);
    packet_addr = get_write_ptr(packet_cb_id);

    {
        DeviceZoneScopedN("R2-PACK-DATA");
        tt_memmove<true, false, false, 0>(packet_addr, get_read_ptr(cb_r1_result_l), payload_size_bytes);
        tt_memmove<true, false, false, 0>(
            packet_addr + payload_size_bytes, get_read_ptr(cb_r1_result_s), aligned_page_size);
        tt_memmove<true, false, false, 0>(
            packet_addr + payload_size_bytes + aligned_page_size, get_read_ptr(cb_r1_result_m), aligned_page_size);
    }

    // NOTE: We do NOT pop cb_r1_result_* here!
    // The compute kernel ALSO needs cb_r1_result_* as input for R2 reduction.
    // Compute will pop them as part of sdpa_reduce.
    // This is a shared-read pattern: both writer and compute read from the same CB,
    // and only compute pops (since it's the "real" consumer for R2 computation).

    // Build fused packet for R2 - goes to R2 neighbor
    // This is a DIFFERENT device than R1 neighbor!
    // Same symmetric shard placement assumption as R1 - see comment above.
    const uint64_t r2_neighbor_dst_noc = get_noc_addr(current_core_x, current_core_y, r2_neighbor_dst_addr);
    const uint64_t r2_neighbor_sem_noc = get_noc_addr(current_core_x, current_core_y, r2_neighbor_sem_addr);

    {
        DeviceZoneScopedN("R2-MUX-SEND");
        r2_header->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{r2_neighbor_dst_noc, r2_neighbor_sem_noc, 1, false},
            align(total_payload_size, l1_alignment));

        // Send R2 via R2 mux (immediate - mux already connected!)
        // r2_mux.wait_for_empty_write_slot();
        // atleast 1 buffer in the mux, and this is the first packet being sent.
        r2_mux.send_payload_without_header_non_blocking_from_address(packet_addr, total_payload_size);
        r2_mux.send_payload_flush_non_blocking_from_address((uint32_t)r2_header, packet_header_size_bytes);
    }

    cb_push_back(packet_cb_id, 1);

    // ==========================================================================
    // PHASE 4: Disconnect muxes (overlaps with R2 compute on TRISC)
    // ==========================================================================
    // Disconnect immediately after sends complete - R2 compute is running in parallel
    // on TRISC, so this disconnect overhead is hidden.
    {
        DeviceZoneScopedN("MUX-DISCONNECT");
        tt::tt_fabric::fabric_client_disconnect(r1_mux);
        tt::tt_fabric::fabric_client_disconnect(r2_mux);
    }
    // ==========================================================================
    // PHASE 5: Terminate muxes (overlaps with R2 compute on TRISC)
    // ==========================================================================
    // Mux termination runs while compute may still be finishing FINAL-NORMALIZE.
    // This masks the termination overhead.

    {
        DeviceZoneScopedN("MUX-TERMINATE");
        // Signal termination for R1 mux
        if (r1_is_term_master) {
            auto* term_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r1_term_sync_addr);
            noc_semaphore_wait(term_ptr, r1_num_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(r1_mux_x, r1_mux_y, r1_mux_term_addr);
        } else {
            uint64_t dest = safe_get_noc_addr(r1_term_master_x, r1_term_master_y, r1_term_sync_addr, 0);
            noc_semaphore_inc(dest, 1);
            noc_async_atomic_barrier();
        }

        // Signal termination for R2 mux
        if (r2_is_term_master) {
            auto* term_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r2_term_sync_addr);
            noc_semaphore_wait(term_ptr, r2_num_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(r2_mux_x, r2_mux_y, r2_mux_term_addr);
        } else {
            uint64_t dest = safe_get_noc_addr(r2_term_master_x, r2_term_master_y, r2_term_sync_addr, 0);
            noc_semaphore_inc(dest, 1);
            noc_async_atomic_barrier();
        }
    }
    noc_async_write_barrier();
    DPRINT << "writer terminated muxes" << ENDL();
}
