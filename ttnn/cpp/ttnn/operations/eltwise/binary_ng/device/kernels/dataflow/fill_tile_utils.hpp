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
    auto* word_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    uint32_t packed_exp = (word_ptr[0] & 0xFF) * 0x01010101u;
    uint32_t packed_data = (word_ptr[16] & 0xFF) * 0x01010101u;

    // Fill 64 exponent bytes = 16 uint32_t words (unrolled by 4)
    for (uint32_t i = 0; i < 16; i += 4) {
        word_ptr[i] = packed_exp;
        word_ptr[i + 1] = packed_exp;
        word_ptr[i + 2] = packed_exp;
        word_ptr[i + 3] = packed_exp;
    }
    // Fill 1024 data bytes = 256 uint32_t words (unrolled by 4, direct indexing)
    for (uint32_t i = 16; i < 272; i += 4) {
        word_ptr[i] = packed_data;
        word_ptr[i + 1] = packed_data;
        word_ptr[i + 2] = packed_data;
        word_ptr[i + 3] = packed_data;
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

    // Pack row-0 exponents into words and write all 4 faces at once
    uint32_t packed_exp0 = static_cast<uint32_t>(byte_ptr[0]) * 0x01010101u;
    uint32_t packed_exp1 = static_cast<uint32_t>(byte_ptr[16]) * 0x01010101u;
    word_ptr[0] = packed_exp0;
    word_ptr[1] = packed_exp0;
    word_ptr[2] = packed_exp0;
    word_ptr[3] = packed_exp0;
    word_ptr[4] = packed_exp1;
    word_ptr[5] = packed_exp1;
    word_ptr[6] = packed_exp1;
    word_ptr[7] = packed_exp1;
    word_ptr[8] = packed_exp0;
    word_ptr[9] = packed_exp0;
    word_ptr[10] = packed_exp0;
    word_ptr[11] = packed_exp0;
    word_ptr[12] = packed_exp1;
    word_ptr[13] = packed_exp1;
    word_ptr[14] = packed_exp1;
    word_ptr[15] = packed_exp1;

    // Cache row-0 data into registers to avoid repeated volatile reads
    uint32_t f0_w0 = word_ptr[16], f0_w1 = word_ptr[17];
    uint32_t f1_w0 = word_ptr[48], f1_w1 = word_ptr[49];

    // Fill face 0 rows 1-15, then face 2 all 16 rows (both from face 0's row 0)
    uint32_t dst = 18;
    for (uint32_t row = 1; row < 16; ++row, dst += 2) {
        word_ptr[dst] = f0_w0;
        word_ptr[dst + 1] = f0_w1;
    }
    dst = 80;
    for (uint32_t row = 0; row < 16; ++row, dst += 2) {
        word_ptr[dst] = f0_w0;
        word_ptr[dst + 1] = f0_w1;
    }

    // Fill face 1 rows 1-15, then face 3 all 16 rows (both from face 1's row 0)
    dst = 50;
    for (uint32_t row = 1; row < 16; ++row, dst += 2) {
        word_ptr[dst] = f1_w0;
        word_ptr[dst + 1] = f1_w1;
    }
    dst = 112;
    for (uint32_t row = 0; row < 16; ++row, dst += 2) {
        word_ptr[dst] = f1_w0;
        word_ptr[dst + 1] = f1_w1;
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
    auto* word_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    // --- Face pair 0: face0 (left) -> face1 (right) ---
    // Replicate column 0 across all 16 columns in left face data (4 words per row)
    uint32_t row_word = 16;  // left face data starts at word 16
    for (uint32_t row = 0; row < 16; ++row) {
        uint32_t col0_val = word_ptr[row_word] & 0xFF;
        uint32_t packed = col0_val | (col0_val << 8) | (col0_val << 16) | (col0_val << 24);
        word_ptr[row_word] = packed;
        word_ptr[row_word + 1] = packed;
        word_ptr[row_word + 2] = packed;
        word_ptr[row_word + 3] = packed;
        row_word += 4;
    }
    // Copy left face exponents to right face (words 0-3 -> 4-7)
    word_ptr[4] = word_ptr[0];
    word_ptr[5] = word_ptr[1];
    word_ptr[6] = word_ptr[2];
    word_ptr[7] = word_ptr[3];
    // Copy left face data to right face (words 16-79 -> 80-143)
    for (uint32_t i = 0; i < 64; ++i) {
        word_ptr[80 + i] = word_ptr[16 + i];
    }

    // --- Face pair 1: face2 (left) -> face3 (right) ---
    // Replicate column 0 across all 16 columns in left face data (4 words per row)
    row_word = 144;  // left face data starts at word 144
    for (uint32_t row = 0; row < 16; ++row) {
        uint32_t col0_val = word_ptr[row_word] & 0xFF;
        uint32_t packed = col0_val | (col0_val << 8) | (col0_val << 16) | (col0_val << 24);
        word_ptr[row_word] = packed;
        word_ptr[row_word + 1] = packed;
        word_ptr[row_word + 2] = packed;
        word_ptr[row_word + 3] = packed;
        row_word += 4;
    }
    // Copy left face exponents to right face (words 8-11 -> 12-15)
    word_ptr[12] = word_ptr[8];
    word_ptr[13] = word_ptr[9];
    word_ptr[14] = word_ptr[10];
    word_ptr[15] = word_ptr[11];
    // Copy left face data to right face (words 144-207 -> 208-271)
    for (uint32_t i = 0; i < 64; ++i) {
        word_ptr[208 + i] = word_ptr[144 + i];
    }
}

// Reads the very first row of the CB and fills the entire tile with the same row.
// Tile is assumed to be in BFP8 format (Bfp8 or Bfp8_b).
// Row broadcast: top faces (0,1) have row 0 data; bottom faces (2,3) are zero.
//   -> Replicate row 0 within top faces, then copy top faces to bottom faces.
FORCE_INLINE void fill_tile_with_first_row_bfp8(uint32_t cb_id) {
    auto* byte_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(get_write_ptr(cb_id));
    auto* word_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    // Pack row-0 exponents into words and write all 4 faces at once
    uint32_t packed_exp0 = static_cast<uint32_t>(byte_ptr[0]) * 0x01010101u;
    uint32_t packed_exp1 = static_cast<uint32_t>(byte_ptr[16]) * 0x01010101u;
    word_ptr[0] = packed_exp0;
    word_ptr[1] = packed_exp0;
    word_ptr[2] = packed_exp0;
    word_ptr[3] = packed_exp0;
    word_ptr[4] = packed_exp1;
    word_ptr[5] = packed_exp1;
    word_ptr[6] = packed_exp1;
    word_ptr[7] = packed_exp1;
    word_ptr[8] = packed_exp0;
    word_ptr[9] = packed_exp0;
    word_ptr[10] = packed_exp0;
    word_ptr[11] = packed_exp0;
    word_ptr[12] = packed_exp1;
    word_ptr[13] = packed_exp1;
    word_ptr[14] = packed_exp1;
    word_ptr[15] = packed_exp1;

    // Cache row-0 data into registers to avoid repeated volatile reads
    uint32_t f0_w0 = word_ptr[16], f0_w1 = word_ptr[17];
    uint32_t f0_w2 = word_ptr[18], f0_w3 = word_ptr[19];
    uint32_t f1_w0 = word_ptr[80], f1_w1 = word_ptr[81];
    uint32_t f1_w2 = word_ptr[82], f1_w3 = word_ptr[83];

    // Fill face 0 rows 1-15, then face 2 all 16 rows (both from face 0's row 0)
    uint32_t dst = 20;
    for (uint32_t row = 1; row < 16; ++row, dst += 4) {
        word_ptr[dst] = f0_w0;
        word_ptr[dst + 1] = f0_w1;
        word_ptr[dst + 2] = f0_w2;
        word_ptr[dst + 3] = f0_w3;
    }
    dst = 144;
    for (uint32_t row = 0; row < 16; ++row, dst += 4) {
        word_ptr[dst] = f0_w0;
        word_ptr[dst + 1] = f0_w1;
        word_ptr[dst + 2] = f0_w2;
        word_ptr[dst + 3] = f0_w3;
    }

    // Fill face 1 rows 1-15, then face 3 all 16 rows (both from face 1's row 0)
    dst = 84;
    for (uint32_t row = 1; row < 16; ++row, dst += 4) {
        word_ptr[dst] = f1_w0;
        word_ptr[dst + 1] = f1_w1;
        word_ptr[dst + 2] = f1_w2;
        word_ptr[dst + 3] = f1_w3;
    }
    dst = 208;
    for (uint32_t row = 0; row < 16; ++row, dst += 4) {
        word_ptr[dst] = f1_w0;
        word_ptr[dst + 1] = f1_w1;
        word_ptr[dst + 2] = f1_w2;
        word_ptr[dst + 3] = f1_w3;
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
