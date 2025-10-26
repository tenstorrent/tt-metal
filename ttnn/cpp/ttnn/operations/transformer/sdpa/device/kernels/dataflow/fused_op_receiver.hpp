// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "debug/assert.h"
#include <array>

struct RingSDPAOpReceiver {
    constexpr static uint32_t num_directions = 2;
    bool wait_for_op_signal = 0;

    uint32_t ring_size = 0;
    uint32_t ring_index = 0;

    std::array<volatile tt_l1_ptr uint32_t*, num_directions> signal_op_semaphore_addr_ptrs = {};
    std::array<uint32_t, num_directions> received_inputs = {};
    std::array<uint32_t, num_directions> expected_inputs = {};
    uint32_t curr_dir = 1;
    uint32_t curr_transfer_idx = 0;

    bool initialized = false;

    RingSDPAOpReceiver() {}

    RingSDPAOpReceiver(bool wait_for_op_signal, uint32_t& rt_args_idx) : wait_for_op_signal(wait_for_op_signal) {
        // Runtime args
        this->ring_size = get_arg_val<uint32_t>(rt_args_idx++);
        this->ring_index = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t forward_writes_expected = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t backward_writes_expected = get_arg_val<uint32_t>(rt_args_idx++);

        this->expected_inputs[0] = backward_writes_expected;
        this->expected_inputs[1] = forward_writes_expected;
        this->received_inputs[0] = 0;
        this->received_inputs[1] = 0;

        if (this->wait_for_op_signal) {
            // First semaphore is AllGather's BWD semaphore. It belongs to direction 1.
            this->signal_op_semaphore_addr_ptrs[1] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
            // Second is AllGather's FWD semaphore. It belongs to direction 0.
            this->signal_op_semaphore_addr_ptrs[0] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
        }

        /**
         * 1: Data coming from backward_device -> this_device
         * 0: Data coming from forward_device -> this_device
         */
        this->curr_dir = 0;  // Start backward
        this->curr_transfer_idx = 0;

        this->initialized = true;
    }

    uint32_t get_next_ring_id_and_sync() {
        ASSERT(this->initialized);
        // Behave differently for first iteration and subsequent iterations

        uint32_t this_direction_inputs = this->received_inputs[this->curr_dir];
        uint32_t sender_ring_id;
        uint32_t sem_wait_val;
        if (this->curr_transfer_idx == 0) {
            // First iteration, waiting on local slice
            sender_ring_id = this->ring_index;
            sem_wait_val = 1;
        } else {
            this->received_inputs[this->curr_dir] += 1;
            if (this->curr_dir == 1) {
                // receiving from the forward direction. go backwards by that many targets
                sender_ring_id =
                    (this->ring_index - this->received_inputs[this->curr_dir] + this->ring_size) % this->ring_size;
                sem_wait_val = this->received_inputs[this->curr_dir];
            } else {
                // receiving from backward direction. go forward by that many targets
                sender_ring_id = (this->ring_index + this->received_inputs[this->curr_dir]) % this->ring_size;
                sem_wait_val = this->received_inputs[this->curr_dir] + 1;
            }
        }

        // Wait for a sempaphore signal to start processing the tensor slice

        if (this->wait_for_op_signal) {
            noc_semaphore_wait_min(this->signal_op_semaphore_addr_ptrs[this->curr_dir], sem_wait_val);
        }

        if (this->curr_transfer_idx == 0) {
            if (this->expected_inputs[this->curr_dir] == 0) {
                // On first transfer, only switch directions if there are no inputs in forward direction
                this->curr_dir = 1 - this->curr_dir;
            }
        } else {
            // Flip direction if we haven't received all inputs in that direction
            uint32_t next_dir = 1 - this->curr_dir;
            if (this->received_inputs[next_dir] < this->expected_inputs[next_dir]) {
                this->curr_dir = next_dir;
            }
        }

        this->curr_transfer_idx++;

        return sender_ring_id;
    }
};
