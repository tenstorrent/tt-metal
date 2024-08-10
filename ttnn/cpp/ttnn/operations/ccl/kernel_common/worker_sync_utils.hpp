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
    volatile uint32_t* master_l1_semaphore_addr,
    const uint32_t num_slaves,
    const uint64_t* remote_slave_l1_semaphore_addrs,
    const uint64_t remote_op_l1_semaphore_addr) {

    // Wait for all the slaves to finish their work
    noc_semaphore_wait(master_l1_semaphore_addr,  num_slaves);
    DPRINT << "MASTER SYNCED WITH SLAVES" << ENDL();

    // Send signal to op
    noc_semaphore_inc(remote_op_l1_semaphore_addr, 1);
    DPRINT << "MASTER SIGNALED REMOTE OP" << ENDL();

    // Clear the master semaphore, so that it can be used again
    noc_semaphore_set(master_l1_semaphore_addr, 0);

    // Clear the slave semaphores, so that they can continue processing
    for (uint32_t i = 0; i < num_slaves; i++) {
        noc_semaphore_inc(remote_slave_l1_semaphore_addrs[i], 1);
        DPRINT << "MASTER CLEAREED A SLAVE SEMAPHORE" << ENDL();
    }
}


// Called by the slave worker to synchronize with the master worker
FORCE_INLINE void slave_sync_master(
    volatile uint32_t* slave_l1_semaphore_addr,
    const uint64_t remote_master_l1_semaphore_addr) {

    // Signal the master that the slave has finished its work
    noc_semaphore_inc(remote_master_l1_semaphore_addr, 1);
    DPRINT << "SLAVE SYNCED WITH MASTER" << ENDL();

    // Wait for the master to signal that this slave is ready to continue
    noc_semaphore_wait(slave_l1_semaphore_addr, 1);
    DPRINT << "SLAVE SEMAPHORE CLEARED BY MASTER" << ENDL();

    // Clear the slave semaphore, so that it can be used again
    noc_semaphore_set(slave_l1_semaphore_addr, 0);
}


FORCE_INLINE bool is_master(uint32_t master_x, uint32_t master_y, uint32_t worker_x, uint32_t worker_y) {
    return master_x == worker_x && master_y == worker_y;
}

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

struct OpSignaler {
    // Compile time args
    uint32_t num_workers_to_sync;
    uint32_t curr_worker_index;
    uint32_t worker_sync_sem_addr;

    // Runtime args
    uint32_t* workers_noc_coords; // Worker NOC coordinates [x1, y1, x2, y2...]
    uint32_t op_worker_noc_x;
    uint32_t op_worker_noc_y;
    uint32_t signal_op_sem_addr;

    uint32_t master_worker_noc_x;
    uint32_t master_worker_noc_y;
    uint32_t curr_worker_noc_x;
    uint32_t curr_worker_noc_y;
    uint32_t curr_worker_is_master;

    volatile tt_l1_ptr uint32_t* curr_worker_l1_semaphore_addr_ptr;
    uint64_t worker_sem_noc_addrs[10]; // First one is for master
    uint64_t signal_op_sem_noc_addr;

    bool initialized = false;

    OpSignaler() {}

    OpSignaler(
        uint32_t num_workers_to_sync,
        uint32_t curr_worker_index,
        uint32_t worker_sync_sem_addr,
        uint32_t* workers_noc_coords,
        uint32_t op_worker_noc_x,
        uint32_t op_worker_noc_y,
        uint32_t signal_op_sem_addr)
        : num_workers_to_sync(num_workers_to_sync),
          curr_worker_index(curr_worker_index),
          worker_sync_sem_addr(worker_sync_sem_addr),
          workers_noc_coords(workers_noc_coords),
          op_worker_noc_x(op_worker_noc_x),
          op_worker_noc_y(op_worker_noc_y),
          signal_op_sem_addr(signal_op_sem_addr) {

        // Get the remote sem addresses to signal the op
        this->signal_op_sem_noc_addr = get_noc_addr(this->op_worker_noc_x, this->op_worker_noc_y, this->signal_op_sem_addr);

        this->master_worker_noc_x = this->workers_noc_coords[0];
        this->master_worker_noc_y = this->workers_noc_coords[1];
        this->curr_worker_noc_x = this->workers_noc_coords[this->curr_worker_index * 2];
        this->curr_worker_noc_y = this->workers_noc_coords[this->curr_worker_index * 2 + 1];
        this->curr_worker_is_master = is_master(this->master_worker_noc_x, this->master_worker_noc_y, this->curr_worker_noc_x, this->curr_worker_noc_y);

        this->curr_worker_l1_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(this->worker_sync_sem_addr);

        // Convert sem addresses into remote sem addresses
        if (this->curr_worker_is_master) { // If cur is master, skip doing the conversion for the master slot
            for (uint32_t i = 1; i < this->num_workers_to_sync; i++) {
                this->worker_sem_noc_addrs[i] = get_noc_addr(this->workers_noc_coords[i * 2], this->workers_noc_coords[i * 2 + 1], this->worker_sync_sem_addr);
            }
        } else { // If cur is slave, only do conversion for the master slot
            worker_sem_noc_addrs[0] = get_noc_addr(this->master_worker_noc_x, this->master_worker_noc_y, this->worker_sync_sem_addr);
        }

        this->initialized = true;
    }

    void synchronize() {
        ASSERT(this->initialized);

        if (this->curr_worker_is_master) {
            master_sync_slaves(this->curr_worker_l1_semaphore_addr_ptr, this->num_workers_to_sync - 1, this->worker_sem_noc_addrs + 1, this->signal_op_sem_noc_addr);
        } else {
            slave_sync_master(this->curr_worker_l1_semaphore_addr_ptr, this->worker_sem_noc_addrs[0]);
        }
    }

};
