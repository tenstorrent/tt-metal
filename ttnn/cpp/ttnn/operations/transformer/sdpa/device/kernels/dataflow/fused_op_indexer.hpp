// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/debug/assert.h"
#include "ring_utils.hpp"

struct RingSDPAOpIndexer {
    RingIdSequencer seq;
    bool initialized = false;

private:
    void initialize(
        uint32_t ring_size, uint32_t ring_index, uint32_t forward_writes_expected, uint32_t backward_writes_expected) {
        seq = RingIdSequencer(ring_index, ring_size, backward_writes_expected, forward_writes_expected);
        initialized = true;
    }

public:
    RingSDPAOpIndexer() = default;

    RingSDPAOpIndexer(
        uint32_t ring_size, uint32_t ring_index, uint32_t forward_writes_expected, uint32_t backward_writes_expected) {
        initialize(ring_size, ring_index, forward_writes_expected, backward_writes_expected);
    }

    RingSDPAOpIndexer(uint32_t& rt_args_idx) {
        uint32_t ring_size = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t ring_index = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t forward_writes_expected = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t backward_writes_expected = get_arg_val<uint32_t>(rt_args_idx++);

        rt_args_idx += 2;  // Skip the semaphore addresses

        initialize(ring_size, ring_index, forward_writes_expected, backward_writes_expected);
    }

    uint32_t get_next_ring_id_and_sync() {
        ASSERT(initialized);
        return seq.get_next_ring_id([](uint32_t, uint32_t) {});
    }
};
