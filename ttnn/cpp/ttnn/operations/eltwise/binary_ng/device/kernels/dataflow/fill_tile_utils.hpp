// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"

// Fills one full tile of bfloat16 with a scalar value
// Scalar is assumed to be a 16-bit value double packed into a u32
FORCE_INLINE void fill_with_val_bfloat16(uint32_t cb_id, uint32_t packed_scalar) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    // 1024 is the number of elements in a full tile, but since the scalar is packed into a u32,
    // each iteration writes 2 elements, hence the division by 2
    for (uint32_t i = 0; i < 512; ++i) {
        ptr[i] = packed_scalar;
    }
}

// Reads the very first element of the CB and fills the entire tile with that value.
// Tile is assumed to have 16-bit elements
FORCE_INLINE void fill_tile_with_first_element_bfloat16(uint32_t cb_id) {
    auto* read_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    const uint16_t first_elem = read_ptr[0];
    const uint32_t packed_first_elem = first_elem << 16 | first_elem;

    // Since all elements in the tile are the same, we can ignore the faces and assume the entire
    // tile is contiguous in memory.
    auto* write_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    // TODO: should I fill one face like this and then use noc to fill the rest?
    for (uint32_t i = 0; i < 512; ++i) {
        write_ptr[i] = packed_first_elem;
    }
}

// Reads the very first row of the CB and fills the entire tile with the same row.
// Tile is assumed to have 16-bit elements.
FORCE_INLINE void fill_tile_with_first_row_bfloat16(uint32_t cb_id) {
    // Here we have to account for the fact that a tile consists of 4 16x16 faces.
    // So we have to fill faces 0 and 2 with the first row of face 0, and faces 1 and 3
    // with the first row of face 1.
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    uint32_t row_offset = 8;  // start at second row since first row is source
    uint32_t num_rows = 15;

    // iterate over face pairs (0,1) and (2,3)
    for(uint32_t k = 0, face_offset = 0; k < 2; ++k, face_offset += 256) {
        for (uint32_t row = 0; row < num_rows; ++row) {
            uint32_t dst_offset = face_offset + row_offset;
            for (uint32_t col = 0; col < 8; ++col) {
                ptr[dst_offset + col] = ptr[col];             // left face
                ptr[dst_offset + col + 128] = ptr[col + 128]; // right face
            }
            row_offset += 8;
        }
        row_offset = 0;
        num_rows = 16;
    }
}

// Reads the very first column of the CB and fills the entire tile with the same column.
// Tile is assumed to have 16-bit elements.
FORCE_INLINE void fill_tile_with_first_column_bfloat16(uint32_t cb_id) {
    // Here we have to account for the fact that a tile consists of 4 16x16 faces.
    // So we have to fill faces 0 and 1 with the first column of face 0, and faces 2 and 3
    // with the first column of face 2.
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));

    constexpr uint32_t num_rows = 16;

    // iterate over face pairs (0,1) and (2,3)
    for (uint32_t k = 0, face_offset = 0; k < 2; ++k, face_offset += 512) {
        for (uint32_t row = 0, row_offset = 0; row < num_rows; ++row, row_offset += 16) {
            uint32_t dst_offset = face_offset + row_offset;
            auto src_val = ptr[dst_offset];

            ptr[dst_offset + 256] = src_val; // first column of right face
            for (uint32_t col = 1; col < 16; ++col) {
                ptr[dst_offset + col] = src_val;       // left face
                ptr[dst_offset + col + 256] = src_val; // right face
            }
        }
    }
}
