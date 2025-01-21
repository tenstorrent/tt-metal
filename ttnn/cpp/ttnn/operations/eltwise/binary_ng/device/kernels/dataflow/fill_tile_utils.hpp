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

template <uint32_t ElementsV, class ScalarT>
FORCE_INLINE void fill_with_val(uint32_t cb_id, ScalarT scalar) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr ScalarT*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < ElementsV; ++i) {
        ptr[i] = scalar;
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

// Reads the very first element of the CB and fills the entire tile with that value.
// Tile is assumed to have 32-bit elements (float32 or int32).
template <typename T>
FORCE_INLINE void fill_tile_with_first_element(uint32_t cb_id) {
    auto* read_ptr = reinterpret_cast<volatile tt_l1_ptr T*>(get_write_ptr(cb_id));
    const T first_elem = read_ptr[0];

    auto* write_ptr = reinterpret_cast<volatile tt_l1_ptr T*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 1024; ++i) {
        write_ptr[i] = first_elem;
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

// Reads the very first row of the CB and fills the entire tile with the same row.
// Tile is assumed to have 32-bit elements (float32/int32).
FORCE_INLINE void fill_tile_with_first_row(uint32_t cb_id) {
    // Tile with 4 faces (16x16) and 32-bit elements
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    uint32_t row_offset = 16;  // Start at the second row (offset by 16 elements)
    uint32_t num_rows = 15;    // 15 rows to fill per face

    // Iterate over face pairs (0,1) and (2,3)
    for (uint32_t k = 0, face_offset = 0; k < 2; ++k, face_offset += 512) {  // Offset 512 = 256 elements x 2 faces
        for (uint32_t row = 0; row < num_rows; ++row) {
            uint32_t dst_offset = face_offset + row_offset;
            for (uint32_t col = 0; col < 16; ++col) {
                ptr[dst_offset + col] = ptr[col];              // left face
                ptr[dst_offset + col + 256] = ptr[col + 256];  // right face
            }
            row_offset += 16;  // Move to the next row (16 elements per row)
        }
        row_offset = 0;  // Reset for the next face pair
        num_rows = 16;   // Process all rows for the next face pair
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

// Reads the very first column of the CB and fills the entire tile with the same column.
// Tile is assumed to have 32-bit elements (float32/int32).
FORCE_INLINE void fill_tile_with_first_column(uint32_t cb_id) {
    // Tile with 4 faces (16x16) and 32-bit elements
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    constexpr uint32_t num_rows = 16;             // Number of rows per face
    constexpr uint32_t face_row_stride = 16;      // Elements per row
    constexpr uint32_t face_size = 256;           // Total elements per face (16x16)
    constexpr uint32_t face_offset_stride = 512;  // Total elements per pair of faces (2x16x16)

    // Iterate over face pairs (0,1) and (2,3)
    for (uint32_t k = 0, face_offset = 0; k < 2; ++k, face_offset += face_offset_stride) {
        for (uint32_t row = 0, row_offset = 0; row < num_rows; ++row, row_offset += face_row_stride) {
            uint32_t left_dst_offset = face_offset + row_offset;      // Left face (0 or 2)
            uint32_t right_dst_offset = left_dst_offset + face_size;  // Right face (1 or 3)

            // Read the first column value for the current row from the left face
            auto src_val = ptr[left_dst_offset];

            for (uint32_t col = 0; col < face_row_stride; ++col) {
                ptr[left_dst_offset + col] = src_val;   // left face
                ptr[right_dst_offset + col] = src_val;  // right face
            }
        }
    }
}
