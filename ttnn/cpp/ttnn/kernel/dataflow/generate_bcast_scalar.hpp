// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"

// W-bcast scalar
// Tile is assumed to have 16-bit elements
// Scalar is assumed to be a 16-bit value double packed into a u32
FORCE_INLINE void generate_bcast_col_scalar(CircularBuffer cb, uint32_t scalar) {
    const uint16_t scalar_val = scalar >> 16;
    cb.reserve_back(1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb.get_write_ptr());
    for (int k = 0; k < 4; k += 2) {
        uint32_t idx = k << 8;
        for (int j = 0; j < 256; j += 16) {
            ptr[idx + j] = scalar_val;
        }
    }
    cb.push_back(1);
}

// H-bcast scalar
// Tile is assumed to have 16-bit elements
// Scalar is assumed to be a 16-bit value double packed into a u32
FORCE_INLINE void generate_bcast_row_scalar(CircularBuffer cb, uint32_t scalar) {
    cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
    for (int k = 0; k < 2; ++k) {
        uint32_t idx = k << 7;
        for (int j = 0; j < 8; ++j) {
            ptr[idx + j] = scalar;
        }
    }
    cb.push_back(1);
}

// HW-bcast scalar
// Tile is assumed to have 16-bit elements
// Scalar is assumed to be a 16-bit value double packed into a u32
FORCE_INLINE void generate_bcast_unary_scalar(CircularBuffer cb, uint32_t scalar) {
    const uint32_t scalar_val = scalar >> 16;
    cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
    ptr[0] = scalar_val;
    cb.push_back(1);
}
