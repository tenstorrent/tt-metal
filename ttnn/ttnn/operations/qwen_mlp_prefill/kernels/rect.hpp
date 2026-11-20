// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "named_args.hpp"

struct Rect {
    uint32_t x_start;
    uint32_t y_start;
    uint32_t x_end;
    uint32_t y_end;
    uint32_t cores;

    void write(const Noc& noc, uint32_t src, uint32_t dst, uint32_t bytes) const {
        noc.async_write_multicast(
            CoreLocalMem<uint32_t>(src),
            MulticastEndpoint{},
            bytes,
            cores,
            {},
            {.noc_x_start = x_start, .noc_y_start = y_start, .noc_x_end = x_end, .noc_y_end = y_end, .addr = dst},
            false);
    }

    void increment(const Noc& noc, Semaphore<>& semaphore, uint32_t value) const {
        for (uint32_t x = x_start; x <= x_end; x++) {
            for (uint32_t y = y_start; y <= y_end; y++) {
                semaphore.up(noc, x, y, value);
            }
        }
    }
};

#define RECT(name)                                                                                      \
    Rect {                                                                                              \
        CT_ARG(name "_x_start"), CT_ARG(name "_y_start"), CT_ARG(name "_x_end"), CT_ARG(name "_y_end"), \
            CT_ARG(name "_cores")                                                                       \
    }

inline uint32_t arena_base(uint32_t cb_id) { return CircularBuffer(cb_id).get_write_ptr(); }
