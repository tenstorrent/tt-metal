// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/dataflow_api.h"

// Tile is assumed to have 16-bit elements
// Scaler is assumed to be a 16-bit value double packed into a u32
FORCE_INLINE void generate_mm_scaler(DataflowBuffer cb, const uint32_t scaler) {
    cb.reserve_back(1);

    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());

    // Fill tile with zeros
    Noc noc;
    noc.async_write_zeros(cb, 2048);
    noc.write_zeros_l1_barrier();

    uint32_t single_packed_scalar = scaler & 0xFFFF;
    for (int i = 0; i < 128; i += 8) {
        ptr[i] = single_packed_scalar;
    }
    for (int i = 256; i < 384; i += 8) {
        ptr[i] = single_packed_scalar;
    }

    cb.push_back(1);
}
