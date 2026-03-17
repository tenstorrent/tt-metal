// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "ring_utils.hpp"
#include <array>

struct RingSDPAOpReceiver {
    RingIdSequencer seq;
    bool wait_for_op_signal = false;
    std::array<volatile tt_l1_ptr uint32_t*, 2> signal_op_semaphore_addr_ptrs = {};
    bool initialized = false;

    RingSDPAOpReceiver() {}

    RingSDPAOpReceiver(bool wait_for_op_signal, uint32_t& rt_args_idx) : wait_for_op_signal(wait_for_op_signal) {
        uint32_t ring_size = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t ring_index = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t forward_writes_expected = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t backward_writes_expected = get_arg_val<uint32_t>(rt_args_idx++);

        if (this->wait_for_op_signal) {
            // First semaphore is AllGather's BWD semaphore. It belongs to direction 1.
            signal_op_semaphore_addr_ptrs[1] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
            // Second is AllGather's FWD semaphore. It belongs to direction 0.
            signal_op_semaphore_addr_ptrs[0] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
        }

        seq = RingIdSequencer(ring_index, ring_size, backward_writes_expected, forward_writes_expected);
        initialized = true;
    }

    uint32_t get_next_ring_id_and_sync() {
        ASSERT(initialized);
        return seq.get_next_ring_id([&](uint32_t dir, uint32_t val) {
            if (this->wait_for_op_signal) {
                // noc_semaphore_wait_min(this->signal_op_semaphore_addr_ptrs[dir], val);
            }
        });
    }
};
