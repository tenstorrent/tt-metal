// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/debug/assert.h"
#include "ring_utils.hpp"
#include <array>

struct RingSDPAOpReceiver {
    RingIdSequencer seq;
    bool wait_for_op_signal = false;
    std::array<uint32_t, 2> signal_op_semaphore_ids = {0, 0};
    bool initialized = false;

    // Even-ring split-forwarding: the diametric shard arrives split across both links and is signaled on both
    bool split_forwarding_enabled = false;
    uint32_t split_shard_id = 0;
    uint32_t split_second_half_wait = 0;

    // Words push_ring_sdpa_fused_op_rt_args() always pushes, versus the number this constructor
    // consumes when wait_for_op_signal is false (it skips the two all-gather semaphore ids and the
    // three split-forwarding words). A kernel that reads its own runtime args after a non-waiting
    // receiver must step over the difference; see ring_joint_writer.cpp.
    static constexpr uint32_t kPushedRtArgCount = 9;
    static constexpr uint32_t kConsumedRtArgCountNoWait = 4;

    RingSDPAOpReceiver() {}

    RingSDPAOpReceiver(bool wait_for_op_signal, uint32_t& rt_args_idx) : wait_for_op_signal(wait_for_op_signal) {
        uint32_t ring_size = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t ring_index = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t forward_writes_expected = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t backward_writes_expected = get_arg_val<uint32_t>(rt_args_idx++);

        if (this->wait_for_op_signal) {
            // First semaphore is AllGather's BWD semaphore. It belongs to direction 1.
            signal_op_semaphore_ids[1] = get_arg_val<uint32_t>(rt_args_idx++);
            // Second is AllGather's FWD semaphore. It belongs to direction 0.
            signal_op_semaphore_ids[0] = get_arg_val<uint32_t>(rt_args_idx++);
            split_forwarding_enabled = get_arg_val<uint32_t>(rt_args_idx++) == 1;
            split_shard_id = get_arg_val<uint32_t>(rt_args_idx++);
            split_second_half_wait = get_arg_val<uint32_t>(rt_args_idx++);
        }

        seq = RingIdSequencer(ring_index, ring_size, backward_writes_expected, forward_writes_expected);
        initialized = true;
    }

    uint32_t get_next_ring_id_and_sync() {
        ASSERT(initialized);
        uint32_t ring_id = seq.get_next_ring_id([&](uint32_t dir, uint32_t val) {
            if (this->wait_for_op_signal) {
                Semaphore<>(this->signal_op_semaphore_ids[dir]).wait_min(val);
            }
        });
        // The split shard's second half is the forward chain's final arrival, on semaphore_ids[0] —
        // the semaphore that also carries the local-slice pre-signal (hence the +2 in the threshold;
        // see push_ring_sdpa_fused_op_rt_args).
        if (this->wait_for_op_signal && this->split_forwarding_enabled && ring_id == this->split_shard_id) {
            Semaphore<>(this->signal_op_semaphore_ids[0]).wait_min(this->split_second_half_wait);
        }
        return ring_id;
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
