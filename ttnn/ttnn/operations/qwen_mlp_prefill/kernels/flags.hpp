// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "ring.hpp"

// Monotonic counters that peer cores write into this core's L1, one word per writer.
struct FlagBoard {
    static constexpr uint32_t ready = 0;
    static constexpr uint32_t credit = 3;
    static constexpr uint32_t words = 6;
    static constexpr uint32_t scratch_words = 4;

    uint32_t base;
    uint32_t scratch_first;
    uint32_t posts;

    volatile tt_l1_ptr uint32_t* word(uint32_t index) const {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + index * CT_ARG("flag_bytes"));
    }

    void clear() const {
        for (uint32_t i = 0; i < words; i++) {
            *word(i) = 0;
        }
    }

    void wait_min(uint32_t index, uint32_t value) const {
        while (*word(index) < value) {
        }
    }

    void post(const Noc& noc, uint32_t peer_x, uint32_t peer_y, uint32_t index, uint32_t value) {
        const uint32_t slot = scratch_first + (posts++ % scratch_words);
        if (posts % scratch_words == 0) {
            noc.async_writes_flushed();
        }
        *word(slot) = value;
        noc.async_write(
            CoreLocalMem<uint32_t>(base + slot * CT_ARG("flag_bytes")),
            UnicastEndpoint{},
            CT_ARG("flag_bytes"),
            {},
            {.noc_x = peer_x, .noc_y = peer_y, .addr = base + index * CT_ARG("flag_bytes")});
    }
};
