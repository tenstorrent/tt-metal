// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

// Device roles
enum MeshRole : uint32_t { MESH_LEAF = 0, MESH_ROOT3 = 1, MESH_ROOT2 = 2, MESH_ROOT1 = 3 };

void kernel_main() {
    // Compile-time args
    constexpr uint32_t device_role = get_compile_time_arg_val(0);
    constexpr uint32_t local_cb = get_compile_time_arg_val(1);
    constexpr uint32_t received_cb = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(3);

    DPRINT << "READER: role=" << device_role << " num_tiles=" << num_tiles << ENDL();

    // Senders don't receive anything - they just read local data
    if constexpr (device_role == MESH_LEAF) {
        // For senders: local_cb is already populated (in-place on input shard)
        // Just push to indicate data is ready
        DPRINT << "READER: LEAF pushing local_cb" << ENDL();
        cb_push_back(local_cb, num_tiles);
        DPRINT << "READER: LEAF done" << ENDL();
        return;
    }

    // Runtime args for receivers (semaphore addresses only, each round receives 1 shard)
    size_t arg_idx = 0;
    const uint32_t recv_sem_round1 = get_arg_val<uint32_t>(arg_idx++);

    uint32_t recv_sem_round2 = 0;
    if constexpr (device_role == MESH_ROOT2 || device_role == MESH_ROOT1) {
        recv_sem_round2 = get_arg_val<uint32_t>(arg_idx++);
    }

    uint32_t recv_sem_round3 = 0;
    if constexpr (device_role == MESH_ROOT1) {
        recv_sem_round3 = get_arg_val<uint32_t>(arg_idx++);
    }

    DPRINT << "READER: sem1=0x" << HEX() << recv_sem_round1 << " sem2=0x" << recv_sem_round2 << " sem3=0x"
           << recv_sem_round3 << ENDL();

    // Push local data to compute (local_cb is in-place on input shard)
    DPRINT << "READER: Pushing local_cb" << ENDL();
    cb_push_back(local_cb, num_tiles);

    // === Round 1: Wait for shard from sender ===
    volatile tt_l1_ptr uint32_t* recv_sem1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_round1);

    DPRINT << "READER: Waiting for round1 sem (cur=" << DEC() << *recv_sem1_ptr << ")" << ENDL();
    noc_semaphore_wait_min(recv_sem1_ptr, 1);
    DPRINT << "READER: Round1 received!" << ENDL();

    // Push received data to compute
    cb_push_back(received_cb, num_tiles);

    noc_semaphore_set(recv_sem1_ptr, 0);

    if constexpr (device_role == MESH_ROOT2 || device_role == MESH_ROOT1) {
        // === Round 2: Wait for result from ROOT3 ===
        volatile tt_l1_ptr uint32_t* recv_sem2_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_round2);

        DPRINT << "READER: Waiting for round2 sem (cur=" << DEC() << *recv_sem2_ptr << ")" << ENDL();
        noc_semaphore_wait_min(recv_sem2_ptr, 1);
        DPRINT << "READER: Round2 received!" << ENDL();

        cb_push_back(received_cb, num_tiles);

        noc_semaphore_set(recv_sem2_ptr, 0);
    }

    if constexpr (device_role == MESH_ROOT1) {
        // === Round 3: Wait for result from ROOT2 ===
        volatile tt_l1_ptr uint32_t* recv_sem3_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_round3);

        DPRINT << "READER: Waiting for round3 sem (cur=" << DEC() << *recv_sem3_ptr << ")" << ENDL();
        noc_semaphore_wait_min(recv_sem3_ptr, 1);
        DPRINT << "READER: Round3 received!" << ENDL();

        cb_push_back(received_cb, num_tiles);

        noc_semaphore_set(recv_sem3_ptr, 0);
    }

    DPRINT << "READER: Done" << ENDL();
}
