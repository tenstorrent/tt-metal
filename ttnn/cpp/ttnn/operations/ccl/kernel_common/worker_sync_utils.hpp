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
    const uint32_t fused_op_sem_addr,

    bool wait_for_start_signal) {

    // Wait for all the slaves to finish their work
    volatile tt_l1_ptr uint32_t* master_l1_semaphore_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sync_sem_addr);
    noc_semaphore_wait(master_l1_semaphore_addr,  num_workers_to_sync - 1 + (uint32_t)wait_for_start_signal);
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

    // Params for start signal
    bool wait_for_start_signal;
    bool send_start_signal;
    uint32_t* start_signal_receiver_core_noc;
    uint32_t start_signal_receiver_sem_addr;

    bool initialized = false;

    OpSignaler() {}

    OpSignaler(uint32_t& rt_args_idx) {

        // Runtime args
        this->num_workers_to_sync = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t curr_worker_index = get_arg_val<uint32_t>(rt_args_idx++);
        this-> worker_sync_sem_addr = get_semaphore(get_arg_val<uint32_t>(rt_args_idx++));
        this->workers_noc_coords = (uint32_t*)get_arg_addr(increment_arg_idx(rt_args_idx, this->num_workers_to_sync * 2)); // Skip over the number of workers

        this->num_fused_op_cores_to_signal = get_arg_val<uint32_t>(rt_args_idx++);
        this->signal_op_cores_noc_coords = (uint32_t*)get_arg_addr(increment_arg_idx(rt_args_idx, this->num_fused_op_cores_to_signal * 2));
        this->signal_op_sem_addr = get_semaphore(get_arg_val<uint32_t>(rt_args_idx++));


        this->wait_for_start_signal = get_arg_val<uint32_t>(rt_args_idx++);
        this->send_start_signal = get_arg_val<uint32_t>(rt_args_idx++);
        if (this->send_start_signal) {
            this->start_signal_receiver_core_noc = (uint32_t*)get_arg_addr(increment_arg_idx(rt_args_idx, 2));
            this->start_signal_receiver_sem_addr = get_semaphore(get_arg_val<uint32_t>(rt_args_idx++));
        }

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
                this->signal_op_sem_addr,

                this->wait_for_start_signal
            );

            // Once start signal is received, no need to wait for it again
            this->wait_for_start_signal = false;

            if (this->send_start_signal) {
                uint64_t remote_master_l1_semaphore_addr = get_noc_addr(
                    this->start_signal_receiver_core_noc[0],
                    this->start_signal_receiver_core_noc[1],
                    this->start_signal_receiver_sem_addr
                );
                noc_semaphore_inc(remote_master_l1_semaphore_addr, 1);

                // Once start signal is sent, no need to send it again
                this->send_start_signal = false;
            }
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


struct MatmulOpReceiver {
    uint32_t num_tensor_slices;
    uint32_t num_directions = 2; // ASSUMPTION: Always 2 directions

    bool wait_for_op_signal;
    uint32_t num_transfers;
    uint32_t ring_size;
    uint32_t tensor_slice_shape_width; // In tiles
    uint32_t output_page_offset;
    uint32_t last_output_page_offset;

    uint32_t num_blocks;
    uint32_t num_blocks_per_slice;

    // Used to track internal state
    uint32_t ring_idxs[2];
    uint32_t start_page_idxs[2];
    bool is_clockwise_dirs[2];
    volatile tt_l1_ptr uint32_t* signal_op_semaphore_addr_ptrs[2];
    uint32_t curr_dir;
    uint32_t curr_transfer_idx;


    bool initialized = false;

    MatmulOpReceiver() {}

    MatmulOpReceiver(
        bool wait_for_op_signal,
        uint32_t& rt_args_idx,
        uint32_t num_blocks,
        uint32_t tiles_per_block // Across the same dimension as tensor_slice_shape_width
    ) : wait_for_op_signal(wait_for_op_signal),
        num_blocks(num_blocks)
    {

        // Runtime args
        this->num_transfers = get_arg_val<uint32_t>(rt_args_idx++);
        this->ring_size = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t start_ring_index = get_arg_val<uint32_t>(rt_args_idx++);
        this->tensor_slice_shape_width = get_arg_val<uint32_t>(rt_args_idx++);
        this->output_page_offset = get_arg_val<uint32_t>(rt_args_idx++);
        this->last_output_page_offset = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t is_clockwise_direction = get_arg_val<uint32_t>(rt_args_idx++);

        if (this->wait_for_op_signal) {
            this->signal_op_semaphore_addr_ptrs[0] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
            this->signal_op_semaphore_addr_ptrs[1] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
        }

        this->num_tensor_slices = this->num_transfers * this->num_directions;

        // Setup internal states for bi-direction
        this->ring_idxs[0] = start_ring_index;
        this->ring_idxs[1] = start_ring_index;

        this->start_page_idxs[0] = this->ring_idxs[0] * this->output_page_offset;
        this->start_page_idxs[1] = this->ring_idxs[1] * this->output_page_offset;

        this->is_clockwise_dirs[0] = is_clockwise_direction;
        this->is_clockwise_dirs[1] = !is_clockwise_direction;

        this->num_blocks_per_slice = this->tensor_slice_shape_width / tiles_per_block;
        ASSERT(this->num_tensor_slices * this->num_blocks_per_slice == this->num_blocks);

        this->curr_dir = is_clockwise_direction ? 1 : 0; // Anti-clockwise direction is the first since it has local slice
        this->curr_transfer_idx = 0;

        this->initialized = true;
    }


    void update_current_block_start_tile_id(
        const uint32_t& block_idx,
        uint32_t& curr_block_start_tile_id,
        const uint32_t& tensor_start_tile_id
    ) {
        ASSERT(this->initialized);

        if (block_idx % this->num_blocks_per_slice == 0) { // Aligned to the start of a tensor slice

            if (this->curr_transfer_idx != 0) { // Skip update for local slice

                // Update the start page idx of the tensor slice in curr_direction
                advance_start_page_idx(
                    this->start_page_idxs[this->curr_dir],
                    this->ring_idxs[this->curr_dir],
                    this->ring_size,
                    this->is_clockwise_dirs[this->curr_dir],
                    this->output_page_offset,
                    this->last_output_page_offset
                );
            }

            // Use the new start page idx to find the start tile id of the current tensor slice
            curr_block_start_tile_id = tensor_start_tile_id + this->start_page_idxs[this->curr_dir];

            // Index of the current tensor slice in a certain direction
            uint32_t tensor_slice_cnt = (this->curr_transfer_idx) / this->num_directions;

            // Wait for a sempaphore signal to start processing the tensor slice
            if (this->wait_for_op_signal) {
                noc_semaphore_wait_min(this->signal_op_semaphore_addr_ptrs[this->curr_dir], tensor_slice_cnt + 1);
            }

            // Update the relevant internal states
            this->curr_transfer_idx++;
            this->curr_dir = !this->curr_dir; // Change direction
        }
    }

    uint32_t align_to_slice_and_sync(uint32_t block_idx, uint32_t sender_id) {
        ASSERT(this->initialized);

        // Align the id to the start of the tensor slice in order of processing from all gather
        uint32_t block_id = this->ring_idxs[this->curr_dir];

        if (block_idx % this->num_blocks_per_slice == 0) { // Aligned to the start of a tensor slice

            if (this->curr_transfer_idx != 0) { // Skip update for local slice
                // Change direction
                this->curr_dir = !this->curr_dir;

                // Update the start page idx of the tensor slice in curr_direction
                // We only want to know the update for the ring index
                advance_start_page_idx(
                    this->start_page_idxs[this->curr_dir],
                    this->ring_idxs[this->curr_dir],
                    this->ring_size,
                    this->is_clockwise_dirs[this->curr_dir],
                    this->output_page_offset,
                    this->last_output_page_offset
                );
            }

            // Update the alignment
            block_id = this->ring_idxs[this->curr_dir];

            // Wait for a sempaphore signal to start processing the tensor slice
            if (this->wait_for_op_signal && block_id == sender_id) {
                uint32_t tensor_slice_cnt = (this->curr_transfer_idx) / this->num_directions;
                noc_semaphore_wait_min(this->signal_op_semaphore_addr_ptrs[this->curr_dir], tensor_slice_cnt + 1);
            }

            this->curr_transfer_idx++;
        }

        return block_id;
    }
};
