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

    RingSDPAOpReceiver() {}

    RingSDPAOpReceiver(bool wait_for_op_signal, uint32_t& rt_args_idx) : wait_for_op_signal(wait_for_op_signal) {
        uint32_t ring_size = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t ring_index = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t forward_writes_expected = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t backward_writes_expected = get_arg_val<uint32_t>(rt_args_idx++);

        // Read the whole pushed block either way, so rt_args_idx lands on the caller's own args in
        // both modes. First semaphore is AllGather's BWD (direction 1), second its FWD (direction 0).
        const uint32_t bwd_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
        const uint32_t fwd_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
        const uint32_t split_forwarding = get_arg_val<uint32_t>(rt_args_idx++);
        const uint32_t split_shard = get_arg_val<uint32_t>(rt_args_idx++);
        const uint32_t split_wait = get_arg_val<uint32_t>(rt_args_idx++);
        if (this->wait_for_op_signal) {
            signal_op_semaphore_ids[1] = bwd_semaphore_id;
            signal_op_semaphore_ids[0] = fwd_semaphore_id;
            split_forwarding_enabled = split_forwarding == 1;
            split_shard_id = split_shard;
            split_second_half_wait = split_wait;
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
