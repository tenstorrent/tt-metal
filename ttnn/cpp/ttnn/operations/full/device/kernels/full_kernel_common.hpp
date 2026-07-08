// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"

union value {
    float f;
    uint32_t u;
};
constexpr uint32_t onepage = 1;

inline void zero_buffer(uint32_t cb_id, uint32_t bytes) {
    Noc noc;
    CircularBuffer cb(cb_id);
    noc.async_write_zeros(cb, bytes);
    noc.write_zeros_l1_barrier();
}
