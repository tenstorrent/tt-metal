// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "debug/assert.h"
#include "debug/dprint.h"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"

// Called by the master worker to synchronize with the slave workers
FORCE_INLINE void master_sync_slaves(

    /* Used to get slave worker's sem addrs */
    const uint32_t num_workers_to_sync,
    const uint32_t* worker_noc_coords,
    const uint32_t worker_sync_sem_addr,

    /* Used to signal the remote op */
    const uint32_t num_fused_op_cores_to_signal,
    const uint32_t* fused_op_cores_noc_coords,
    const uint32_t fused_op_sem_addr) {

    // Wait for all the slaves to finish their work
    volatile tt_l1_ptr uint32_t* master_l1_semaphore_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sync_sem_addr);
    noc_semaphore_wait(master_l1_semaphore_addr,  num_workers_to_sync - 1);
    // DPRINT << "MASTER SYNCED WITH SLAVES" << ENDL();

    // Send signal to op
    for (uint32_t i = 0; i < num_fused_op_cores_to_signal; i++) {
        uint64_t remote_fused_op_l1_semaphore_addr = get_noc_addr(fused_op_cores_noc_coords[i * 2], fused_op_cores_noc_coords[i * 2 + 1], fused_op_sem_addr);
        noc_semaphore_inc(remote_fused_op_l1_semaphore_addr, 1);
    }
    // DPRINT << "MASTER SIGNALED REMOTE OP" << ENDL();

    // Clear the master semaphore, so that it can be used again
    noc_semaphore_set(master_l1_semaphore_addr, 0);

    // Clear the slave semaphores, so that they can continue processing
    for (uint32_t i = 1; i < num_workers_to_sync; i++) { // Skip the first set of coords because they are for master worker
        uint64_t remote_slave_l1_sem_addr = get_noc_addr(worker_noc_coords[i * 2], worker_noc_coords[i * 2 + 1], worker_sync_sem_addr);
        noc_semaphore_inc(remote_slave_l1_sem_addr, 1);
        // DPRINT << "MASTER CLEAREED A SLAVE SEMAPHORE" << ENDL();
    }
}


// Called by the slave worker to synchronize with the master worker
FORCE_INLINE void slave_sync_master(
    const uint32_t* worker_noc_coords,
    const uint32_t worker_sync_sem_addr) {

    // Signal the master that the slave has finished its work
    uint64_t remote_master_l1_semaphore_addr = get_noc_addr(worker_noc_coords[0], worker_noc_coords[1], worker_sync_sem_addr);
    noc_semaphore_inc(remote_master_l1_semaphore_addr, 1);
    // DPRINT << "SLAVE SYNCED WITH MASTER" << ENDL();

    // Wait for the master to signal that this slave is ready to continue
    volatile tt_l1_ptr uint32_t* slave_l1_semaphore_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sync_sem_addr);
    noc_semaphore_wait(slave_l1_semaphore_addr, 1);
    // DPRINT << "SLAVE SEMAPHORE CLEARED BY MASTER" << ENDL();

    // Clear the slave semaphore, so that it can be used again
    noc_semaphore_set(slave_l1_semaphore_addr, 0);
}


FORCE_INLINE bool is_master(uint32_t master_x, uint32_t master_y, uint32_t worker_x, uint32_t worker_y) {
    return master_x == worker_x && master_y == worker_y;
}

uint32_t increment_arg_idx(uint32_t& arg_idx, uint32_t num_args=1) {
    uint32_t old_arg_idx = arg_idx;
    arg_idx += num_args;
    return old_arg_idx;
}

// Used to signal an operation that it can start processing data, resulting in overlapping
struct OpSignaler {
    uint32_t num_workers_to_sync;
    uint32_t* workers_noc_coords; // Worker NOC coordinates [x1, y1, x2, y2...], first one is for master
    uint32_t worker_sync_sem_addr;

    uint32_t num_fused_op_cores_to_signal;
    uint32_t* signal_op_cores_noc_coords;
    uint32_t signal_op_sem_addr;
    uint32_t curr_worker_is_master;
    bool initialized = false;

    OpSignaler() {}

    OpSignaler(
        uint32_t num_workers_to_sync,
        uint32_t curr_worker_index,
        uint32_t worker_sync_sem_id,
        uint32_t& rt_args_idx) :
        num_workers_to_sync(num_workers_to_sync) {

        this-> worker_sync_sem_addr = get_semaphore(worker_sync_sem_id);

        // Runtime args
        this->workers_noc_coords = (uint32_t*)get_arg_addr(increment_arg_idx(rt_args_idx, this->num_workers_to_sync * 2)); // Skip over the number of workers

        this->num_fused_op_cores_to_signal = get_arg_val<uint32_t>(rt_args_idx++);
        this->signal_op_cores_noc_coords = (uint32_t*)get_arg_addr(increment_arg_idx(rt_args_idx, this->num_fused_op_cores_to_signal * 2));
        this->signal_op_sem_addr = get_semaphore(get_arg_val<uint32_t>(rt_args_idx++));

        uint32_t master_worker_noc_x = this->workers_noc_coords[0];
        uint32_t master_worker_noc_y = this->workers_noc_coords[1];
        uint32_t curr_worker_noc_x = this->workers_noc_coords[curr_worker_index * 2];
        uint32_t curr_worker_noc_y = this->workers_noc_coords[curr_worker_index * 2 + 1];
        this->curr_worker_is_master = is_master(master_worker_noc_x, master_worker_noc_y, curr_worker_noc_x, curr_worker_noc_y);

        this->initialized = true;
    }

    void synchronize_workers_and_signal_op() {
        ASSERT(this->initialized);

        if (this->curr_worker_is_master) {
            master_sync_slaves(
                this->num_workers_to_sync,
                this->workers_noc_coords,
                this->worker_sync_sem_addr,

                this->num_fused_op_cores_to_signal,
                this->signal_op_cores_noc_coords,
                this->signal_op_sem_addr
            );
        } else {
            slave_sync_master(this->workers_noc_coords, this->worker_sync_sem_addr);
        }
    }

};

// Used by datacopy kernel
FORCE_INLINE void advance_start_page_idx(
    uint32_t& start_page_idx,
    uint32_t& curr_ring_index,
    const uint32_t ring_size,
    const uint32_t is_clockwise_direction,
    const uint32_t output_page_offset,
    const uint32_t last_output_page_offset) {

    if (is_clockwise_direction) {
        bool is_wraparound_ring_index = curr_ring_index == 0;
        if (is_wraparound_ring_index) {
            start_page_idx += last_output_page_offset;
            curr_ring_index = ring_size - 1;
        } else {
            start_page_idx -= output_page_offset;
            curr_ring_index--;
        }
    } else {
        // counter clockwise direction
        bool is_wraparound_ring_index = curr_ring_index == ring_size - 1;
        if (is_wraparound_ring_index) {
            start_page_idx -= last_output_page_offset;
            curr_ring_index = 0;
        } else {
            start_page_idx += output_page_offset;
            curr_ring_index++;
        }
    }

}
