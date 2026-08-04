// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/debug/assert.h"
#include "exp_ring_utils.hpp"

struct RingSDPAOpIndexer {
    RingIdSequencer seq;
    bool initialized = false;

    RingSDPAOpIndexer() {}

    // RT arg layout: ring_size, ring_index, direction (3 values)
    RingSDPAOpIndexer(uint32_t& rt_args_idx) {
        uint32_t ring_size = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t ring_index = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t direction = get_arg_val<uint32_t>(rt_args_idx++);

        seq = RingIdSequencer(ring_index, ring_size, direction);
        initialized = true;
    }

    uint32_t get_next_ring_id_and_sync() {
        ASSERT(initialized);
        return seq.get_next_ring_id([](uint32_t, uint32_t) {});
    }
};
