// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "ring_utils.hpp"

struct RingSDPAOpReceiver {
    RingIdSequencer seq;
    bool wait_for_op_signal = false;
    volatile tt_l1_ptr uint32_t* signal_op_semaphore_addr_ptr = nullptr;
    bool initialized = false;

    RingSDPAOpReceiver() {}

    // RT arg layout (4 values when wait_for_op_signal=true, 3 when false):
    //   ring_size, ring_index, direction, [semaphore_id]
    RingSDPAOpReceiver(bool wait_for_op_signal, uint32_t& rt_args_idx) : wait_for_op_signal(wait_for_op_signal) {
        uint32_t ring_size = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t ring_index = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t direction = get_arg_val<uint32_t>(rt_args_idx++);

        if (this->wait_for_op_signal) {
            signal_op_semaphore_addr_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
        }

        seq = RingIdSequencer(ring_index, ring_size, direction);
        initialized = true;
    }

    uint32_t get_next_ring_id_and_sync() {
        ASSERT(initialized);
        return seq.get_next_ring_id([&](uint32_t, uint32_t val) {
            if (this->wait_for_op_signal) {
                // noc_semaphore_wait_min(this->signal_op_semaphore_addr_ptr, val);
            }
        });
    }
};
