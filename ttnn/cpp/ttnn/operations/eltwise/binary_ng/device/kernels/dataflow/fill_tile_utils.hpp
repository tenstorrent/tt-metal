// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

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

// Reads the very first element of the CB and fills the entire tile with that value.
// Tile is assumed to be in BFP8 format (Bfp8 or Bfp8_b).
// BFP8 tile layout (32x32 = 1024 elements, 1088 bytes total):
//   Bytes 0-63:    Exponent section (16 uint32_t words = 64 bytes)
//                  4 faces x 16 rows/face, one exponent byte per row of 16 elements
//   Bytes 64-1087: Data section (256 uint32_t words = 1024 bytes)
//                  4 faces x 256 bytes/face, one byte per element
FORCE_INLINE void fill_tile_with_first_element_bfp8(uint32_t cb_id) {
    auto* byte_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(get_write_ptr(cb_id));
    uint8_t exp_val = byte_ptr[0];    // First exponent byte
    uint8_t data_val = byte_ptr[64];  // First data byte (after 64 exponent bytes)

    uint32_t packed_exp =
        (uint32_t)exp_val | ((uint32_t)exp_val << 8) | ((uint32_t)exp_val << 16) | ((uint32_t)exp_val << 24);
    uint32_t packed_data =
        (uint32_t)data_val | ((uint32_t)data_val << 8) | ((uint32_t)data_val << 16) | ((uint32_t)data_val << 24);

    auto* write_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    // Fill 64 exponent bytes = 16 uint32_t words
    for (uint32_t i = 0; i < 16; ++i) {
        write_ptr[i] = packed_exp;
    }
    // Fill 1024 data bytes = 256 uint32_t words (starting at word offset 16)
    for (uint32_t i = 0; i < 256; ++i) {
        write_ptr[16 + i] = packed_data;
    }
}

// Reads the very first element of the CB and fills the entire tile with that value.
// Tile is assumed to be in BFP4 format (Bfp4 or Bfp4_b).
// BFP4 tile layout (32x32 = 1024 elements, 576 bytes total):
//   Bytes 0-63:   Exponent section (16 uint32_t words = 64 bytes)
//                 4 faces x 16 rows/face, one exponent byte per row of 16 elements
//   Bytes 64-575: Data section (128 uint32_t words = 512 bytes)
//                 4 faces x 128 bytes/face, 4 bits per element (2 elements per byte)
// BFP4 nibble packing: element[even] in low nibble (bits 0-3), element[odd] in high nibble (bits 4-7).
FORCE_INLINE void fill_tile_with_first_element_bfp4(uint32_t cb_id) {
    auto* byte_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(get_write_ptr(cb_id));
    uint8_t exp_val = byte_ptr[0];  // First exponent byte

    // Element 0 (the scalar) is in the LOW nibble of the first data byte.
    // Element 1 (padding, typically 0) is in the HIGH nibble.
    // We must pack the scalar into both nibbles so every element gets the same value.
    uint8_t elem0 = byte_ptr[64] & 0x0F;
    uint8_t data_val = (elem0 << 4) | elem0;

    uint32_t packed_exp =
        (uint32_t)exp_val | ((uint32_t)exp_val << 8) | ((uint32_t)exp_val << 16) | ((uint32_t)exp_val << 24);
    uint32_t packed_data =
        (uint32_t)data_val | ((uint32_t)data_val << 8) | ((uint32_t)data_val << 16) | ((uint32_t)data_val << 24);

    auto* write_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    // Fill 64 exponent bytes = 16 uint32_t words
    for (uint32_t i = 0; i < 16; ++i) {
        write_ptr[i] = packed_exp;
    }
    // Fill 512 data bytes = 128 uint32_t words (starting at word offset 16)
    for (uint32_t i = 0; i < 128; ++i) {
        write_ptr[16 + i] = packed_data;
    }
}

// Reads the very first column of the CB and fills the entire tile with the same column.
// Tile is assumed to be in BFP4 format (Bfp4 or Bfp4_b).
// BFP4 tile layout (576 bytes):
//   Exponents: face0 [0..15], face1 [16..31], face2 [32..47], face3 [48..63]
//   Data:      face0 [64..191], face1 [192..319], face2 [320..447], face3 [448..575]
//   Each face data: 16 rows x 8 bytes/row (2 elements per byte, 16 elements per row)
// Face arrangement: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
// Column broadcast: left faces (0,2) have column 0 data; right faces (1,3) are zero.
//   -> Extract column 0 nibble, pack into both nibbles, fill row, then copy left to right face.
FORCE_INLINE void fill_tile_with_first_column_bfp4(uint32_t cb_id) {
    auto* byte_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(get_write_ptr(cb_id));
    auto* word_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    // Process left->right face pairs: (face0->face1) and (face2->face3)
    for (uint32_t pair = 0; pair < 2; ++pair) {
        uint32_t left_exp_word = pair * 8;               // word 0 or 8
        uint32_t right_exp_word = left_exp_word + 4;     // word 4 or 12
        uint32_t left_data_byte = 64 + pair * 256;       // byte 64 or 320
        uint32_t left_data_word = left_data_byte / 4;    // word 16 or 80
        uint32_t right_data_word = left_data_word + 32;  // word 48 or 112

        // 1. For each row, extract column 0 (low nibble of first byte), fill all 8 bytes (2 words)
        for (uint32_t row = 0; row < 16; ++row) {
            uint8_t elem0 = byte_ptr[left_data_byte + row * 8] & 0x0F;
            uint8_t fill_byte = (elem0 << 4) | elem0;
            uint32_t packed = (uint32_t)fill_byte | ((uint32_t)fill_byte << 8) | ((uint32_t)fill_byte << 16) |
                              ((uint32_t)fill_byte << 24);
            uint32_t row_word = left_data_word + row * 2;
            word_ptr[row_word] = packed;
            word_ptr[row_word + 1] = packed;
        }

        // 2. Copy left face exponents to right face (4 words)
        for (uint32_t i = 0; i < 4; ++i) {
            word_ptr[right_exp_word + i] = word_ptr[left_exp_word + i];
        }

        // 3. Copy left face data to right face (32 words)
        for (uint32_t i = 0; i < 32; ++i) {
            word_ptr[right_data_word + i] = word_ptr[left_data_word + i];
        }
    }
}

// Reads the very first row of the CB and fills the entire tile with the same row.
// Tile is assumed to be in BFP4 format (Bfp4 or Bfp4_b).
// Row broadcast: top faces (0,1) have row 0 data; bottom faces (2,3) are zero.
//   -> Replicate row 0 within top faces, then copy top faces to bottom faces.
FORCE_INLINE void fill_tile_with_first_row_bfp4(uint32_t cb_id) {
    auto* byte_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(get_write_ptr(cb_id));
    auto* word_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    // 1. In each top face, replicate row 0 exponent and data across rows 1-15
    for (uint32_t face = 0; face < 2; ++face) {
        uint32_t exp_base = face * 16;             // byte 0 or 16
        uint32_t data_base_word = 16 + face * 32;  // word 16 or 48

        // Replicate row 0 exponent to rows 1-15
        uint8_t row0_exp = byte_ptr[exp_base];
        for (uint32_t row = 1; row < 16; ++row) {
            byte_ptr[exp_base + row] = row0_exp;
        }

        // Replicate row 0 data (2 words = 8 bytes) to rows 1-15
        for (uint32_t row = 1; row < 16; ++row) {
            uint32_t dst_word = data_base_word + row * 2;
            word_ptr[dst_word] = word_ptr[data_base_word];
            word_ptr[dst_word + 1] = word_ptr[data_base_word + 1];
        }
    }

    // 2. Copy top face pair to bottom face pair
    // Exponents: top (words 0-7) -> bottom (words 8-15)
    for (uint32_t i = 0; i < 8; ++i) {
        word_ptr[8 + i] = word_ptr[i];
    }
    // Data: top (words 16-79) -> bottom (words 80-143)
    for (uint32_t i = 0; i < 64; ++i) {
        word_ptr[80 + i] = word_ptr[16 + i];
    }
}

// Reads the very first column of the CB and fills the entire tile with the same column.
// Tile is assumed to be in BFP8 format (Bfp8 or Bfp8_b).
// BFP8 tile layout (1088 bytes):
//   Exponents: face0 [0..15], face1 [16..31], face2 [32..47], face3 [48..63]
//   Data:      face0 [64..319], face1 [320..575], face2 [576..831], face3 [832..1087]
//   Each face data: 16 rows x 16 cols = 256 bytes (1 byte per element, row-major)
// Face arrangement: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
// Column broadcast: left faces (0,2) have column 0 data; right faces (1,3) are zero.
//   -> Replicate column 0 within left faces, then copy left faces to right faces.
FORCE_INLINE void fill_tile_with_first_column_bfp8(uint32_t cb_id) {
    auto* byte_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(get_write_ptr(cb_id));
    auto* word_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    // Process left->right face pairs: (face0->face1) and (face2->face3)
    for (uint32_t pair = 0; pair < 2; ++pair) {
        uint32_t left_exp_word = pair * 8;               // word 0 or 8
        uint32_t right_exp_word = left_exp_word + 4;     // word 4 or 12
        uint32_t left_data_byte = 64 + pair * 512;       // byte 64 or 576
        uint32_t left_data_word = left_data_byte / 4;    // word 16 or 144
        uint32_t right_data_word = left_data_word + 64;  // word 80 or 208

        // 1. Replicate column 0 across columns 1-15 in left face data (4 words per row)
        for (uint32_t row = 0; row < 16; ++row) {
            uint8_t col0_val = byte_ptr[left_data_byte + row * 16];
            uint32_t packed = (uint32_t)col0_val | ((uint32_t)col0_val << 8) | ((uint32_t)col0_val << 16) |
                              ((uint32_t)col0_val << 24);
            uint32_t row_word = left_data_word + row * 4;
            for (uint32_t w = 0; w < 4; ++w) {
                word_ptr[row_word + w] = packed;
            }
        }

        // 2. Copy left face exponents to right face (4 words)
        for (uint32_t i = 0; i < 4; ++i) {
            word_ptr[right_exp_word + i] = word_ptr[left_exp_word + i];
        }

        // 3. Copy left face data to right face (64 words)
        for (uint32_t i = 0; i < 64; ++i) {
            word_ptr[right_data_word + i] = word_ptr[left_data_word + i];
        }
    }
}

// Reads the very first row of the CB and fills the entire tile with the same row.
// Tile is assumed to be in BFP8 format (Bfp8 or Bfp8_b).
// Row broadcast: top faces (0,1) have row 0 data; bottom faces (2,3) are zero.
//   -> Replicate row 0 within top faces, then copy top faces to bottom faces.
FORCE_INLINE void fill_tile_with_first_row_bfp8(uint32_t cb_id) {
    auto* byte_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(get_write_ptr(cb_id));
    auto* word_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    // 1. In each top face, replicate row 0 exponent and data across rows 1-15
    for (uint32_t face = 0; face < 2; ++face) {
        uint32_t exp_base = face * 16;             // byte 0 or 16
        uint32_t data_base_word = 16 + face * 64;  // word 16 or 80

        // Replicate row 0 exponent to rows 1-15
        uint8_t row0_exp = byte_ptr[exp_base];
        for (uint32_t row = 1; row < 16; ++row) {
            byte_ptr[exp_base + row] = row0_exp;
        }

        // Replicate row 0 data (4 words = 16 bytes) to rows 1-15
        for (uint32_t row = 1; row < 16; ++row) {
            uint32_t dst_word = data_base_word + row * 4;
            for (uint32_t w = 0; w < 4; ++w) {
                word_ptr[dst_word + w] = word_ptr[data_base_word + w];
            }
        }
    }

    // 2. Copy top face pair to bottom face pair
    // Exponents: top (words 0-7) -> bottom (words 8-15)
    for (uint32_t i = 0; i < 8; ++i) {
        word_ptr[8 + i] = word_ptr[i];
    }
    // Data: top (words 16-143) -> bottom (words 144-271)
    for (uint32_t i = 0; i < 128; ++i) {
        word_ptr[144 + i] = word_ptr[16 + i];
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
    for (uint32_t k = 0, face_offset = 0; k < 2; ++k, face_offset += 256) {
        for (uint32_t row = 0; row < num_rows; ++row) {
            uint32_t dst_offset = face_offset + row_offset;
            for (uint32_t col = 0; col < 8; ++col) {
                ptr[dst_offset + col] = ptr[col];              // left face
                ptr[dst_offset + col + 128] = ptr[col + 128];  // right face
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

            ptr[dst_offset + 256] = src_val;  // first column of right face
            for (uint32_t col = 1; col < 16; ++col) {
                ptr[dst_offset + col] = src_val;        // left face
                ptr[dst_offset + col + 256] = src_val;  // right face
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
