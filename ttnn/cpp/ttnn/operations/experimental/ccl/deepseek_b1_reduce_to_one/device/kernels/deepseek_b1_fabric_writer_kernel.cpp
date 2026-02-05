// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
// Fabric writer kernel: dedicated fabric cores handle fabric communication.
//
// Uses raw WorkerToFabricEdmSender for fabric connection.
// Uses fused write+atomic_inc packets to send data AND increment destination semaphore.
//
// Dedicated fabric cores don't have their own shard data - they only forward worker packets.
//
// Flow for LEAF / ROOT3 / ROOT2:
//   1. Wait for worker arrivals (workers send pre-assembled packets)
//   2. Forward worker packets via fabric
//
// Flow for ROOT1:
//   No-op - workers handle output via NOC copy

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
    // Compile-time args: role, sizes, CB
    constexpr uint32_t device_role = get_compile_time_arg_val(0);
    constexpr uint32_t num_workers = get_compile_time_arg_val(1);
    constexpr uint32_t payload_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t packet_cb = get_compile_time_arg_val(3);
    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
    constexpr uint32_t slot_size_bytes = packet_header_size_bytes + payload_size_bytes;

    // ROOT1: dedicated fabric core has nothing to do (workers handle output)
    if constexpr (device_role == MESH_ROOT1) {
        return;
    }

    // Runtime args
    size_t arg_idx = 0;
    // Read worker semaphore IDs and convert to L1 addresses
    uint32_t worker_sem_addr[num_workers];
    for (uint32_t i = 0; i < num_workers; i++) {
        worker_sem_addr[i] = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    }
    // Get packet buffer address from CB - this is where workers write their packets
    const uint32_t packet_buffer_addr = get_write_ptr(packet_cb);

    // Build and open fabric connection
    auto fabric_sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    fabric_sender.open();

    // Forward pre-assembled worker packets - wait on each worker's semaphore
    uint32_t slot_base = packet_buffer_addr;
    for (uint32_t worker = 0; worker < num_workers; worker++) {
        uint32_t worker_header_addr = slot_base;
        uint32_t worker_payload_addr = slot_base + packet_header_size_bytes;

        // Wait on this worker's local semaphore (workers increment via NOC)
        volatile tt_l1_ptr uint32_t* worker_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sem_addr[worker]);
        noc_semaphore_wait(worker_sem_ptr, 1);
        noc_semaphore_set(worker_sem_ptr, 0);

        fabric_sender.wait_for_empty_write_slot();
        fabric_sender.send_payload_without_header_non_blocking_from_address(worker_payload_addr, payload_size_bytes);
        fabric_sender.send_payload_flush_blocking_from_address(
            (uint32_t)worker_header_addr, sizeof(PACKET_HEADER_TYPE));

        slot_base += slot_size_bytes;
    }

    fabric_sender.close();

    noc_async_write_barrier();
}
