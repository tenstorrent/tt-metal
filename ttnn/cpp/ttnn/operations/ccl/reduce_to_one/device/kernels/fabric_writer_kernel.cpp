// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
// Fabric writer kernel: bottom cores (row 3) handle fabric communication.
//
// Uses raw WorkerToFabricEdmSender for fabric connection.
// Uses fused write+atomic_inc packets to send data AND increment destination semaphore.
//
// Flow for SENDER / ROOT3 / ROOT2:
//   1. Wait for own data (local_cb for SENDER, output_cb for ROOT)
//   2. Send own shard via fabric
//   3. Wait for worker arrivals (3 workers)
//   4. Forward pre-assembled worker packets
//
// Flow for ROOT1:
//   1. Wait for compute output (in-place), no fabric send

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

// Device roles
enum MeshRole : uint32_t { MESH_LEAF = 0, MESH_ROOT3 = 1, MESH_ROOT2 = 2, MESH_ROOT1 = 3 };

void kernel_main() {
    // Compile-time args
    constexpr uint32_t device_role = get_compile_time_arg_val(0);
    constexpr uint32_t local_cb = get_compile_time_arg_val(1);   // For SENDER: input, for ROOT: output
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);  // For ROOT1: final output CB
    constexpr uint32_t num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t payload_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t num_workers = get_compile_time_arg_val(6);
    constexpr uint32_t packet_cb = get_compile_time_arg_val(7);  // CB index for worker packets
    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    // Runtime args
    size_t arg_idx = 0;
    const uint32_t dst_l1_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dst_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dst_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dst_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_arrival_sem = get_arg_val<uint32_t>(arg_idx++);
    arg_idx++;  // Skip placeholder packet_buffer_addr - we use CB address directly
    const uint32_t slot_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_hops = get_arg_val<uint32_t>(arg_idx++);

    // Get packet buffer address from CB - this is where workers write their packets
    const uint32_t packet_buffer_addr = get_write_ptr(packet_cb);

    // Build raw fabric connection from runtime args (always created, even for ROOT1)
    auto fabric_sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    fabric_sender.open();

    // Destination NOC addresses for fused write+atomic_inc
    uint64_t dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_l1_addr);
    uint64_t dst_sem_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_sem_addr);

    DPRINT << "FABRIC_WRITER: role=" << device_role << " dst_l1=0x" << HEX() << dst_l1_addr << " dst_noc=(" << dst_noc_x
           << "," << dst_noc_y << ")"
           << " dst_sem=0x" << dst_sem_addr << " packet_buf=0x" << packet_buffer_addr << " arrival_sem=0x"
           << worker_arrival_sem << " num_workers=" << DEC() << num_workers << ENDL();

    // Worker arrival semaphore
    volatile tt_l1_ptr uint32_t* arrival_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_arrival_sem);

    if constexpr (device_role == MESH_LEAF || device_role == MESH_ROOT3 || device_role == MESH_ROOT2) {
        // === SENDER/ROOT3/ROOT2: Send own data + forward worker packets ===
        DPRINT << "FABRIC_WRITER: Starting send flow" << ENDL();

        // Allocate header for own shard
        auto own_route_id = PacketHeaderPool::allocate_header_n(1);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* own_header = PacketHeaderPool::header_table[own_route_id].first;

        // Wait for own data (local_cb for SENDER, output_cb for ROOT - passed via local_cb compile arg)
        DPRINT << "FABRIC_WRITER: Waiting for local_cb" << ENDL();
        cb_wait_front(local_cb, num_tiles);
        uint32_t own_data_addr = get_read_ptr(local_cb);
        DPRINT << "FABRIC_WRITER: Got local data at 0x" << HEX() << own_data_addr << ENDL();

        // Set up header for own shard with fused write+atomic_inc
        fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)own_header, num_hops);
        own_header->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, dst_sem_noc_addr, 1, false},
            payload_size_bytes);

        // Send own shard via fabric (payload first, then header)
        fabric_sender.wait_for_empty_write_slot();
        fabric_sender.send_payload_without_header_non_blocking_from_address(own_data_addr, payload_size_bytes);
        fabric_sender.send_payload_flush_non_blocking_from_address(
            reinterpret_cast<uint32_t>(own_header), sizeof(PACKET_HEADER_TYPE));

        cb_pop_front(local_cb, num_tiles);
        DPRINT << "FABRIC_WRITER: Sent own shard via fabric" << ENDL();

        // Forward pre-assembled worker packets
        DPRINT << "FABRIC_WRITER: Starting to forward " << DEC() << num_workers << " worker packets from slot_base=0x"
               << HEX() << packet_buffer_addr << ENDL();
        uint32_t slot_base = packet_buffer_addr;
        for (uint32_t worker = 0; worker < num_workers; worker++) {
            DPRINT << "FABRIC_WRITER: Waiting for worker " << DEC() << worker << " (sem >= " << (worker + 1) << ")"
                   << ENDL();
            noc_semaphore_wait_min(arrival_sem_ptr, worker + 1);

            // Worker's packet is at slot: [header (packet_header_size_bytes)] [payload (payload_size_bytes)]
            uint32_t worker_header_addr = slot_base;
            uint32_t worker_payload_addr = slot_base + packet_header_size_bytes;
            DPRINT << "FABRIC_WRITER: Worker " << worker << " header=0x" << HEX() << worker_header_addr << " payload=0x"
                   << worker_payload_addr << ENDL();

            // Forward: send payload first, then header (header already has fused atomic inc set up)
            fabric_sender.wait_for_empty_write_slot();
            fabric_sender.send_payload_without_header_non_blocking_from_address(
                worker_payload_addr, payload_size_bytes);
            fabric_sender.send_payload_flush_non_blocking_from_address(worker_header_addr, sizeof(PACKET_HEADER_TYPE));

            slot_base += slot_size_bytes;
            DPRINT << "FABRIC_WRITER: Forwarded worker " << DEC() << worker << ENDL();
        }

        // Reset semaphore after all workers have arrived
        noc_semaphore_set(arrival_sem_ptr, 0);
        DPRINT << "FABRIC_WRITER: All workers forwarded, closing fabric" << ENDL();

        fabric_sender.close();

    } else if constexpr (device_role == MESH_ROOT1) {
        // === ROOT1: Output in-place, just sync with compute ===
        // Wait for compute to finish (output is in-place on output tensor)
        cb_wait_front(output_cb, num_tiles);
        cb_pop_front(output_cb, num_tiles);
    }

    noc_async_write_barrier();
}
