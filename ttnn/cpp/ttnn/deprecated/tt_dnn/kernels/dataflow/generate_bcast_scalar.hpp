// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"

// W-bcast scalar
// Tile is assumed to have 16-bit elements
// Scalar is assumed to be a 16-bit value double packed into a u32
FORCE_INLINE void generate_bcast_col_scalar(const uint32_t cb_id, const uint32_t scalar) {
    const uint16_t scalar_val = scalar >> 16;
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    for (int k = 0; k < 4; k += 2) {
        uint32_t idx = k << 8;
        for (int j = 0; j < 256; j += 16) {
            ptr[idx + j] = scalar_val;
        }
    }
    cb_push_back(cb_id, 1);
}

// H-bcast scalar
// Tile is assumed to have 16-bit elements
// Scalar is assumed to be a 16-bit value double packed into a u32
FORCE_INLINE void generate_bcast_row_scalar(const uint32_t cb_id, const uint32_t scalar) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    for (int k = 0; k < 2; ++k) {
        uint32_t idx = k << 7;
        for (int j = 0; j < 8; ++j) {
            ptr[idx + j] = scalar;
        }
    }
    cb_push_back(cb_id, 1);
}

// HW-bcast scalar
// Tile is assumed to have 16-bit elements
// Scalar is assumed to be a 16-bit value double packed into a u32
FORCE_INLINE void generate_bcast_unary_scalar(const uint32_t cb_id, const uint32_t scalar) {
    const uint32_t scalar_val = scalar >> 16;
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    ptr[0] = scalar >> 16;
    cb_push_back(cb_id, 1);
}
