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
    const uint32_t num_workers_to_sync,

    /* Used to get slave worker's sem addrs */
    const uint32_t* worker_noc_coords,
    const uint32_t worker_sync_sem_addr,

    const uint64_t remote_op_l1_semaphore_addr) {

    // Wait for all the slaves to finish their work
    volatile tt_l1_ptr uint32_t* master_l1_semaphore_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sync_sem_addr);
    noc_semaphore_wait(master_l1_semaphore_addr,  num_workers_to_sync - 1);
    // DPRINT << "MASTER SYNCED WITH SLAVES" << ENDL();

    // Send signal to op
    noc_semaphore_inc(remote_op_l1_semaphore_addr, 1);
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
    uint64_t signal_op_sem_noc_addr;
    uint32_t curr_worker_is_master;
    bool initialized = false;

    OpSignaler() {}

    OpSignaler(
        uint32_t num_workers_to_sync,
        uint32_t curr_worker_index,
        uint32_t worker_sync_sem_addr,
        uint32_t& rt_args_idx) :
        num_workers_to_sync(num_workers_to_sync),
        worker_sync_sem_addr(worker_sync_sem_addr) {

        // Runtime args
        this->workers_noc_coords = (uint32_t*)get_arg_addr(increment_arg_idx(rt_args_idx, this->num_workers_to_sync * 2)); // Skip over the number of workers
        uint32_t op_worker_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t op_worker_noc_y = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t signal_op_sem_addr = get_arg_val<uint32_t>(rt_args_idx++);

        // Get the remote sem addresses to signal the op
        this->signal_op_sem_noc_addr = get_noc_addr(op_worker_noc_x, op_worker_noc_y, signal_op_sem_addr);

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
            master_sync_slaves(this->num_workers_to_sync, this->workers_noc_coords, this->worker_sync_sem_addr, this->signal_op_sem_noc_addr);
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

template <typename AddrGen>
FORCE_INLINE void datacopy_read_wrapped_chunk(
    uint32_t& curr_page_idx,
    uint32_t& offset_into_worker_slice,
    ttnn::ccl::coord_t& offset_worker_slice,
    const  ttnn::ccl::coord_t& worker_slice_shape,

    // In tiles for tile layout
    const  ttnn::ccl::coord_t& tensor_shape,
    const  ttnn::ccl::coord_t& tensor_slice_shape,
    const uint32_t cb_id,
    const AddrGen& s,
    const uint32_t num_pages,
    const uint32_t page_size,
    bool& last_page_of_worker,
    uint32_t local_l1_read_addr) {

    ASSERT(last_page_of_worker == false);
    cb_reserve_back(cb_id, num_pages);
    for (uint32_t i = 0; i < num_pages; ++i) {
        noc_async_read_tile(curr_page_idx, s, local_l1_read_addr);

        // Update the curr_page_idx based on how the worker chunks + tensor slice is laid out in global tensor
        advance_worker_global_page_interleaved(
            curr_page_idx, // Updated internally
            offset_into_worker_slice,
            offset_worker_slice,
            worker_slice_shape,
            tensor_slice_shape,
            tensor_shape,
            last_page_of_worker
        );

        local_l1_read_addr += page_size;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}



template <typename AddrGen>
FORCE_INLINE void datacopy_write_wrapped_chunk(
    uint32_t& curr_page_idx,
    uint32_t& offset_into_worker_slice,
    ttnn::ccl::coord_t& offset_worker_slice,
    const  ttnn::ccl::coord_t& worker_slice_shape,

    // In tiles for tile layout
    const  ttnn::ccl::coord_t& tensor_shape,
    const  ttnn::ccl::coord_t& tensor_slice_shape,
    uint32_t cb_id,
    const AddrGen& d,
    const uint32_t num_pages,
    const uint32_t page_size,
    bool& last_page_of_worker,
    uint32_t l1_read_addr) {

    cb_wait_front(cb_id, num_pages);
    for (uint32_t i = 0; i < num_pages; ++i) {
        noc_async_write_tile(curr_page_idx, d, l1_read_addr);

        // Update the curr_page_idx based on how the worker chunks + tensor slice is laid out in global tensor
        advance_worker_global_page_interleaved(
            curr_page_idx, // Updated internally
            offset_into_worker_slice,
            offset_worker_slice,
            worker_slice_shape,
            tensor_slice_shape,
            tensor_shape,
            last_page_of_worker
        );

        l1_read_addr += page_size;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}
