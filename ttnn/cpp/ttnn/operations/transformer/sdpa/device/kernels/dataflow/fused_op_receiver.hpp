// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/debug/assert.h"
#include "ring_utils.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_slice_readiness.hpp"
#include <array>

// slice_readiness: the producer sends progressive slices (bit 1 of the split-forwarding arg agrees). A compile-time
// choice, so consumers without it carry none of its code.
template <bool slice_readiness = false>
struct RingSDPAOpReceiverT {
    RingIdSequencer seq;
    bool wait_for_op_signal = false;
    std::array<uint32_t, 2> signal_op_semaphore_ids = {0, 0};
    bool initialized = false;

    // Even-ring split-forwarding: the diametric shard arrives split across both links and is signaled on both
    bool split_forwarding_enabled = false;
    uint32_t split_shard_id = 0;
    uint32_t split_second_half_wait = 0;

    // Progressive readiness (ring_attention_slice_readiness.hpp): every full remote slice is signaled once per part;
    // get_next_ring_id_and_sync waits for part 0 only, and wait_part waits for later parts as the reader reaches them.
    // Kept per direction: interleaved ring steps (ring_interleave) read one slice from each direction at a time.
    uint32_t last_dir = 0;  // direction of the slice the last get_next_ring_id_and_sync returned
    bool slice_progressive[2] = {false, false};
    uint32_t slice_base[2] = {0, 0};
    uint32_t slice_parts_ready[2] = {0, 0};

    RingSDPAOpReceiverT() {}

    RingSDPAOpReceiverT(bool wait_for_op_signal, uint32_t& rt_args_idx) :
        RingSDPAOpReceiverT(
            wait_for_op_signal, rt_args_idx, [](uint32_t index) { return get_arg_val<uint32_t>(index); }) {}

    template <typename ReadArg>
    RingSDPAOpReceiverT(bool wait_for_op_signal, uint32_t& rt_args_idx, ReadArg read_arg) :
        wait_for_op_signal(wait_for_op_signal) {
        uint32_t ring_size = read_arg(rt_args_idx++);
        uint32_t ring_index = read_arg(rt_args_idx++);
        uint32_t forward_writes_expected = read_arg(rt_args_idx++);
        uint32_t backward_writes_expected = read_arg(rt_args_idx++);

        // Read the whole pushed block either way, so rt_args_idx lands on the caller's own args in
        // both modes. First semaphore is AllGather's BWD (direction 1), second its FWD (direction 0).
        const uint32_t bwd_semaphore_id = read_arg(rt_args_idx++);
        const uint32_t fwd_semaphore_id = read_arg(rt_args_idx++);
        const uint32_t split_forwarding = read_arg(rt_args_idx++);
        const uint32_t split_shard = read_arg(rt_args_idx++);
        const uint32_t split_wait = read_arg(rt_args_idx++);
        if (this->wait_for_op_signal) {
            signal_op_semaphore_ids[1] = bwd_semaphore_id;
            signal_op_semaphore_ids[0] = fwd_semaphore_id;
            split_forwarding_enabled = (split_forwarding & 1) != 0;
            ASSERT(((split_forwarding & 2) != 0) == slice_readiness);
            split_shard_id = split_shard;
            split_second_half_wait = split_wait;
        }

        seq = RingIdSequencer(ring_index, ring_size, backward_writes_expected, forward_writes_expected);
        initialized = true;
    }

    uint32_t get_next_ring_id_and_sync() {
        ASSERT(initialized);
        uint32_t ring_id = seq.get_next_ring_id([&](uint32_t dir, uint32_t val) {
            this->last_dir = dir;
            this->slice_progressive[dir] = false;
            if (this->wait_for_op_signal) {
                if (slice_readiness && val > 0) {
                    // val counts slices in this direction (direction 0 also the local pre-signal); every earlier slice
                    // in the direction is full, so this one's markers start after theirs.
                    const uint32_t pre = dir == 0 ? 1 : 0;
                    this->slice_base[dir] = pre + (val - pre - 1) * ttnn::ring_attention_all_gather::kReadinessParts;
                    this->slice_parts_ready[dir] = 1;
                    this->slice_progressive[dir] = true;
                    Semaphore<>(this->signal_op_semaphore_ids[dir]).wait_min(this->slice_base[dir] + 1);
                } else {
                    Semaphore<>(this->signal_op_semaphore_ids[dir]).wait_min(val);
                }
            }
        });
        // The split shard's second half is the forward chain's final arrival, on semaphore_ids[0], the semaphore that
        // also carries the local-slice pre-signal: its threshold is one signal past the pre-signal and every earlier
        // full slice's markers (push_ring_sdpa_fused_op_rt_args). Its first half arrives whole, with one marker.
        if (this->wait_for_op_signal && this->split_forwarding_enabled && ring_id == this->split_shard_id) {
            this->slice_progressive[this->last_dir] = false;
            Semaphore<>(this->signal_op_semaphore_ids[0]).wait_min(this->split_second_half_wait);
        }
        return ring_id;
    }

    // The remote slice last returned from direction `dir` arrives in parts: wait for `part` before reading a chunk.
    void wait_part(uint32_t dir, uint32_t part) {
        static_assert(slice_readiness);
        if (this->slice_progressive[dir] && part >= this->slice_parts_ready[dir]) {
            Semaphore<>(this->signal_op_semaphore_ids[dir]).wait_min(this->slice_base[dir] + part + 1);
            this->slice_parts_ready[dir] = part + 1;
        }
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

using RingSDPAOpReceiver = RingSDPAOpReceiverT<>;
