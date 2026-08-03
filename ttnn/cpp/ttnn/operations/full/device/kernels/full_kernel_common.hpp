// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_buffer.h"

union value {
    float f;
    uint32_t u;
};
constexpr uint32_t onepage = 1;

// Zeroes the first `bytes` of the caller's dataflow buffer at its current write cursor.
// Takes the buffer the caller already holds rather than a handle: one buffer must be driven by a
// single DataflowBuffer object, so the caller's object is passed through instead of constructing a
// second one for the same FIFO.
inline void zero_buffer(const DataflowBuffer& dfb, uint32_t bytes) {
    Noc noc;
    noc.async_write_zeros(dfb, bytes);
    noc.write_zeros_l1_barrier();
}
