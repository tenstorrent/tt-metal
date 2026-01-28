// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
// Worker writer kernel: non-bottom cores (rows 0,1,2) assemble packets and send to bottom core via NOC.
//
// CB convention (unified across all roles):
//   - source_cb: LEAF uses local_cb (input), others use output_cb (compute output)
//
// Workers:
// 1. Wait for data in source_cb
// 2. Allocate packet header from PacketHeaderPool
// 3. Set up header with fused write+atomic_inc (data + semaphore increment)
// 4. Copy (header, payload) to bottom core's packet buffer slot
// 5. Signal arrival to bottom core
//
// Early exits for ROOT1 device (output is in-place, no sending needed).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

// Device roles
enum MeshRole : uint32_t { MESH_LEAF = 0, MESH_ROOT3 = 1, MESH_ROOT2 = 2, MESH_ROOT1 = 3 };

void kernel_main() {
    // Compile-time args
    constexpr uint32_t device_role = get_compile_time_arg_val(0);
    constexpr uint32_t source_cb = get_compile_time_arg_val(1);  // LEAF: local_cb, others: scratch_cb2
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t payload_size_bytes = get_compile_time_arg_val(3);  // Total payload size
    constexpr uint32_t packet_cb = get_compile_time_arg_val(4);  // CB index for worker packets
    constexpr uint32_t output_cb = get_compile_time_arg_val(5);  // For ROOT1 to wait on compute
    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    // ROOT1 doesn't send - wait for compute to finish writing to source_cb (scratch_cb2), then NOC copy to output
    if constexpr (device_role == MESH_ROOT1) {
        // First round is partial results
        cb_wait_front(source_cb, num_tiles);
        cb_pop_front(source_cb, num_tiles);
        // Wait for compute to push final result to source_cb (scratch_cb2)
        cb_wait_front(source_cb, num_tiles);
        uint32_t src_addr = get_read_ptr(source_cb);
        uint32_t dst_addr = get_write_ptr(output_cb);
        // Local NOC copy from scratch_cb2 to output_cb
        noc_async_write(src_addr, get_noc_addr(dst_addr), payload_size_bytes);
        noc_async_write_barrier();
        cb_pop_front(source_cb, num_tiles);
        return;
    }

    // Runtime args
    uint32_t arg_idx = 0;
    const uint32_t bottom_core_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t bottom_core_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_slot_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t arrival_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t slot_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    // Routing info for packet header
    const uint32_t num_hops = get_arg_val<uint32_t>(arg_idx++);
    // My own NOC coords (same logical position = same physical coords on dest device)
    const uint32_t my_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dst_l1_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dst_sem_addr = get_arg_val<uint32_t>(arg_idx++);  // Destination semaphore for fused atomic inc

    // Get packet buffer address from CB - same address on all cores
    // Workers write to bottom core's packet_cb at this address
    const uint32_t packet_buffer_addr = get_write_ptr(packet_cb);

    // Payload size is already the total (passed as payload_size_bytes from host)
    const uint32_t payload_size = payload_size_bytes;

    // Allocate and set up packet header
    auto route_id = PacketHeaderPool::allocate_header_n(1);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::header_table[route_id].first;

    // Set up routing (linear fabric API)
    fabric_set_unicast_route<false>((tt::tt_fabric::LowLatencyPacketHeader*)packet_header, num_hops);

    // Set up fused NOC write + atomic inc command
    // Destination is the same logical core position on the remote device
    // Same logical position = same physical NOC coords (my_noc_x, my_noc_y)
    uint64_t dst_noc_addr = get_noc_addr(my_noc_x, my_noc_y, dst_l1_addr);
    uint64_t dst_sem_noc_addr = get_noc_addr(my_noc_x, my_noc_y, dst_sem_addr);
    packet_header->to_noc_fused_unicast_write_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, dst_sem_noc_addr, 1, false}, payload_size);

    // Calculate destination slot in bottom core's packet buffer
    uint32_t slot_offset = my_slot_idx * slot_size_bytes;
    uint32_t header_dest_addr = packet_buffer_addr + slot_offset;
    uint32_t payload_dest_addr = header_dest_addr + packet_header_size_bytes;

    uint64_t header_noc_addr = get_noc_addr(bottom_core_noc_x, bottom_core_noc_y, header_dest_addr);
    uint64_t payload_noc_addr = get_noc_addr(bottom_core_noc_x, bottom_core_noc_y, payload_dest_addr);

    // Wait for data in source CB
    cb_wait_front(source_cb, num_tiles);
    uint32_t data_addr = get_read_ptr(source_cb);

    // Send header to bottom core
    noc_async_write(reinterpret_cast<uint32_t>(packet_header), header_noc_addr, packet_header_size_bytes);

    // Send payload to bottom core (right after header)
    noc_async_write(data_addr, payload_noc_addr, payload_size);

    // Signal bottom core by incrementing this worker's semaphore on bottom core
    uint64_t arrival_sem_noc_addr = get_noc_addr(bottom_core_noc_x, bottom_core_noc_y, arrival_sem_addr);
    noc_semaphore_inc(arrival_sem_noc_addr, 1);

    cb_pop_front(source_cb, num_tiles);

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
