// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"

// Tile is assumed to have 16-bit elements
// Scaler is assumed to be a 16-bit value double packed into a u32
FORCE_INLINE void generate_mm_scaler(const uint32_t cb_id, const uint32_t scaler) {
    cb_reserve_back(cb_id, 1);

    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

    // Fill tile with zeros
    Noc noc;
    CircularBuffer cb(cb_id);
    noc.async_write_zeros(cb, 2048);
    noc.write_zeros_l1_barrier();

    uint32_t single_packed_scalar = scaler & 0xFFFF;
    for (int i = 0; i < 128; i += 8) {
        ptr[i] = single_packed_scalar;
    }
    for (int i = 256; i < 384; i += 8) {
        ptr[i] = single_packed_scalar;
    }

    cb_push_back(cb_id, 1);
}
