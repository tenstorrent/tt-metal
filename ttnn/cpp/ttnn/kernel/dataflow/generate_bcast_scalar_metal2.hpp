// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/dataflow_api.h"

// W-bcast scalar
// Tile is assumed to have 16-bit elements
// Scalar is assumed to be a 16-bit value double packed into a u32
//
// 32x32 tiles store 4 faces of 16x16. Tiny tiles (e.g. 1x32) store densely packed
// faces of height min(tile_r, 16) and width 16. Column-broadcast writes the first
// element of each row in the left faces.
FORCE_INLINE void generate_bcast_col_scalar(DataflowBuffer& dfb, uint32_t scalar) {
    const uint16_t scalar_val = scalar >> 16;
    dfb.reserve_back(1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dfb.get_write_ptr());
    const uint32_t num_elements = dfb.get_tile_size() / sizeof(uint16_t);
    constexpr uint32_t face_c = 16;
    constexpr uint32_t tile_c = 32;
    const uint32_t tile_r = num_elements / tile_c;
    const uint32_t face_r = tile_r < 16 ? tile_r : 16;
    const uint32_t face_rows = tile_r / face_r;
    constexpr uint32_t faces_per_row = tile_c / face_c;
    const uint32_t face_size = face_r * face_c;
    for (uint32_t face_row = 0; face_row < face_rows; ++face_row) {
        const uint32_t offset = face_row * faces_per_row * face_size;
        for (uint32_t r = 0; r < face_r; ++r) {
            ptr[offset + r * face_c] = scalar_val;
        }
    }
    dfb.push_back(1);
}

// H-bcast scalar
// Tile is assumed to have 16-bit elements
// Scalar is assumed to be a 16-bit value double packed into a u32
FORCE_INLINE void generate_bcast_row_scalar(DataflowBuffer& dfb, uint32_t scalar) {
    dfb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb.get_write_ptr());
    for (int k = 0; k < 2; ++k) {
        uint32_t idx = k << 7;
        for (int j = 0; j < 8; ++j) {
            ptr[idx + j] = scalar;
        }
    }
    dfb.push_back(1);
}

// HW-bcast scalar
// Tile is assumed to have 16-bit elements
// Scalar is assumed to be a 16-bit value double packed into a u32
FORCE_INLINE void generate_bcast_unary_scalar(DataflowBuffer& dfb, uint32_t scalar) {
    const uint32_t scalar_val = scalar >> 16;
    dfb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb.get_write_ptr());
    ptr[0] = scalar_val;
    dfb.push_back(1);
}
