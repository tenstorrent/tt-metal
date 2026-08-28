// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/debug/assert.h"
#include "api/debug/dprint.h"
#include "ring_utils.hpp"
#include <array>

struct RingSDPAOpReceiver {
    RingIdSequencer seq;
    bool wait_for_op_signal = false;
    std::array<uint32_t, 2> signal_op_semaphore_ids = {0, 0};
    bool initialized = false;

    RingSDPAOpReceiver() {}

    RingSDPAOpReceiver(bool wait_for_op_signal, uint32_t& rt_args_idx) : wait_for_op_signal(wait_for_op_signal) {
        uint32_t ring_size = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t ring_index = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t forward_writes_expected = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t backward_writes_expected = get_arg_val<uint32_t>(rt_args_idx++);

        // Read both semaphore ids: the host always pushes both.
        // First semaphore is AllGather's BWD semaphore (direction 1); second is FWD (direction 0).
        signal_op_semaphore_ids[1] = get_arg_val<uint32_t>(rt_args_idx++);
        signal_op_semaphore_ids[0] = get_arg_val<uint32_t>(rt_args_idx++);

        seq = RingIdSequencer(ring_index, ring_size, backward_writes_expected, forward_writes_expected);
        initialized = true;
    }

    uint32_t get_next_ring_id_and_sync() {
        ASSERT(initialized);
        return seq.get_next_ring_id([&](uint32_t dir, uint32_t val) {
            if (this->wait_for_op_signal) {
                // DEBUG (windowed-gather bring-up): the last "seqwait" with no matching "seqok" is the
                // hanging wait -- it names the direction and the threshold the clamped AG never reached.
                DPRINT("seqwait dir{} want{}\n", (uint32_t)dir, (uint32_t)val);
                Semaphore<>(this->signal_op_semaphore_ids[dir]).wait_min(val);
                DPRINT("seqok dir{} want{}\n", (uint32_t)dir, (uint32_t)val);
            }
        });
    }

    uint32_t get_next_ring_id_and_consume_one_signal() {
        ASSERT(initialized);
        return seq.get_next_ring_id([&](uint32_t dir, uint32_t val) {
            if (this->wait_for_op_signal && val > 0) {
                ASSERT(val == 1);
                Semaphore<>(this->signal_op_semaphore_ids[dir]).down(1);
            }
        });
    }
};
