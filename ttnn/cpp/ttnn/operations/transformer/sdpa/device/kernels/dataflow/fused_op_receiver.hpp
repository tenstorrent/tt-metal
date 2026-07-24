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

    RingSDPAOpReceiver() {}

    RingSDPAOpReceiver(bool wait_for_op_signal, uint32_t& rt_args_idx) : wait_for_op_signal(wait_for_op_signal) {
        uint32_t ring_size = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t ring_index = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t forward_writes_expected = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t backward_writes_expected = get_arg_val<uint32_t>(rt_args_idx++);

        // The host (RingSDPAFusedOpSignaler::push_ring_sdpa_fused_op_rt_args) ALWAYS appends both
        // AllGather semaphore ids, regardless of whether this receiver waits on them. Consume both
        // unconditionally so the runtime-arg stream stays aligned for any args that follow the
        // fused-op block (e.g. the writer's sparse frame_allow words + q_work_bitmap). Previously
        // the reads were gated on wait_for_op_signal, so a wait_for_op_signal=false receiver (the
        // writer) consumed 2 fewer args than the host pushed, shifting all following runtime args
        // by 2 words and corrupting its q_work_bitmap. The semaphores are only *honored* (waited on)
        // in get_next_ring_id_and_sync when wait_for_op_signal is set.
        // First semaphore is AllGather's BWD semaphore. It belongs to direction 1.
        signal_op_semaphore_ids[1] = get_arg_val<uint32_t>(rt_args_idx++);
        // Second is AllGather's FWD semaphore. It belongs to direction 0.
        signal_op_semaphore_ids[0] = get_arg_val<uint32_t>(rt_args_idx++);

        seq = RingIdSequencer(ring_index, ring_size, backward_writes_expected, forward_writes_expected);
        initialized = true;
    }

    uint32_t get_next_ring_id_and_sync() {
        ASSERT(initialized);
        return seq.get_next_ring_id([&](uint32_t dir, uint32_t val) {
            if (this->wait_for_op_signal) {
                Semaphore<>(this->signal_op_semaphore_ids[dir]).wait_min(val);
            }
        });
    }
};
