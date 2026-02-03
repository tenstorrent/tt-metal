// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
// Receiver reader kernel: receives data from fabric and pushes to compute kernel.
//
// On receiving devices (ROOT3, ROOT2, ROOT1):
// 1. Wait for semaphore to be incremented (fused atomic inc signals data arrival)
// 2. Push received data to compute kernel via CB
//
// The data is written directly to received_cb by the fabric (via fused atomic inc packet).
// This kernel just waits for the signal and synchronizes with compute.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

// Device roles
enum MeshRole : uint32_t { MESH_LEAF = 0, MESH_ROOT3 = 1, MESH_ROOT2 = 2, MESH_ROOT1 = 3 };

void kernel_main() {
    // Compile-time args: role, num_tiles, CBs
    constexpr uint32_t device_role = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t local_cb = get_compile_time_arg_val(2);
    constexpr uint32_t received_cb_r1 = get_compile_time_arg_val(3);  // LEAF → ROOT*
    constexpr uint32_t received_cb_r2 = get_compile_time_arg_val(4);  // ROOT3 → ROOT2/ROOT1
    constexpr uint32_t received_cb_r3 = get_compile_time_arg_val(5);  // ROOT2 → ROOT1

    // Runtime args - all 3 semaphore addresses always passed
    size_t arg_idx = 0;
    const uint32_t recv_sem_round1 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t recv_sem_round2 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t recv_sem_round3 = get_arg_val<uint32_t>(arg_idx++);

    if constexpr (device_role == MESH_ROOT3 || device_role == MESH_ROOT2 || device_role == MESH_ROOT1) {
        // Push local data to compute (local_cb is in-place on input shard)
        cb_reserve_back(local_cb, num_tiles);
        cb_push_back(local_cb, num_tiles);

        // === Round 1: Wait for shard from sender (LEAF → ROOT*) ===
        cb_reserve_back(received_cb_r1, num_tiles);
        volatile tt_l1_ptr uint32_t* recv_sem1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_round1);
        noc_semaphore_wait(recv_sem1_ptr, 1);
        noc_semaphore_set(recv_sem1_ptr, 0);
        cb_push_back(received_cb_r1, num_tiles);
    }

    if constexpr (device_role == MESH_ROOT2 || device_role == MESH_ROOT1) {
        // === Round 2: Wait for result from ROOT3 (ROOT3 → ROOT2/ROOT1) ===
        cb_reserve_back(received_cb_r2, num_tiles);
        volatile tt_l1_ptr uint32_t* recv_sem2_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_round2);
        noc_semaphore_wait(recv_sem2_ptr, 1);
        noc_semaphore_set(recv_sem2_ptr, 0);
        cb_push_back(received_cb_r2, num_tiles);
    }

    if constexpr (device_role == MESH_ROOT1) {
        // === Round 3: Wait for result from ROOT2 (ROOT2 → ROOT1) ===
        cb_reserve_back(received_cb_r3, num_tiles);
        volatile tt_l1_ptr uint32_t* recv_sem3_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_round3);
        noc_semaphore_wait(recv_sem3_ptr, 1);
        noc_semaphore_set(recv_sem3_ptr, 0);
        cb_push_back(received_cb_r3, num_tiles);
    }
}
