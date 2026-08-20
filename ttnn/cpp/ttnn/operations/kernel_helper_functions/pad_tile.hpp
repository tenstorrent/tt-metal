// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/constants.hpp>

/**
 * @brief Pads a face within a tile.
 *
 * This function handles padding for a face within a tile. It calculates the appropriate
 * padding for the face based on the unpadded dimensions and applies it.

    Case 1: All elements in face are padded (num_elements_unpadded_w <= face_w_offset)

    num_elements_unpadded_w
            v
    +----------------+----------------+
    | x x x          |                |
    | x x x          |     Face       |
    | x x x          |                |
    +----------------+----------------+
    ^                ^
                face_w_offset

    Case 2: All elements are unpadded (num_elements_unpadded_w >= face_w_offset + FACE_WIDTH)
    +----------------+----------------+
    | x x x x x x x x| x x            |
    | x x Face  x x x| x x            |
    | x x x x x x x x| x x            |
    +----------------+----------------+
    ^                ^
    face_w_offset    face_w_offset + FACE_WIDTH

    Case 3: Some elements are padded (face_w_offset < num_elements_unpadded_w < face_w_offset + FACE_WIDTH)
            num_elements_unpadded_w
                          v
    +----------------+----------------+
    | x x x x x x x x| x x            |
    | x x x x x x x x| x x  Face      |
    | x x x x x x x x| x x            |
    +----------------+----------------+
    ^                ^                ^
                face_w_offset         face_w_offset + FACE_WIDTH

 * @tparam T The type of the elements in the tile
 * @tparam num_elements_unpadded_w Number of unpadded elements in the width dimension
 * @tparam num_elements_unpadded_h Number of unpadded elements in the height dimension
 * @tparam num_faces_w Number of faces in the width dimension
 * @tparam num_faces_h Number of faces in the height dimension
 * @tparam face_h The height index of the face
 * @tparam face_w The width index of the face
 * @param tile_ptr Pointer to the start of the tile
 * @param fill_value The value to use for padding
 */
template <
    typename T,
    uint32_t num_elements_unpadded_w,
    uint32_t num_elements_unpadded_h,
    uint32_t num_faces_w,
    uint32_t num_faces_h,
    uint32_t face_h,
    uint32_t face_w>
void fill_pad_face(T* tile_ptr, T fill_value) {
    using namespace tt::constants;

    // Calculate face offset
    constexpr uint32_t face_offset = (face_h * num_faces_w + face_w) * FACE_HW;
    auto face_ptr = tile_ptr + face_offset;

    // Right padding (width padding)
    constexpr uint32_t face_w_offset = face_w * FACE_WIDTH;
    constexpr uint32_t face_pad_w = (num_elements_unpadded_w <= face_w_offset) ? FACE_WIDTH
                                    : (num_elements_unpadded_w >= face_w_offset + FACE_WIDTH)
                                        ? 0
                                        : face_w_offset + FACE_WIDTH - num_elements_unpadded_w;

    if constexpr (face_pad_w > 0) {
        for (uint32_t row = 0; row < FACE_HEIGHT; ++row) {
            auto row_ptr = face_ptr + row * FACE_WIDTH;
#pragma GCC unroll 32
            for (uint32_t col = FACE_WIDTH - face_pad_w; col < FACE_WIDTH; ++col) {
                row_ptr[col] = fill_value;
            }
        }
    }

    // Bottom padding (height padding)
    constexpr uint32_t face_h_offset = face_h * FACE_HEIGHT;
    constexpr uint32_t face_pad_h = (num_elements_unpadded_h <= face_h_offset) ? FACE_HEIGHT
                                    : (num_elements_unpadded_h >= face_h_offset + FACE_HEIGHT)
                                        ? 0
                                        : face_h_offset + FACE_HEIGHT - num_elements_unpadded_h;

    if constexpr (face_pad_h > 0) {
        for (uint32_t row = FACE_HEIGHT - face_pad_h; row < FACE_HEIGHT; ++row) {
            auto row_ptr = face_ptr + row * FACE_WIDTH;
#pragma GCC unroll 32
            for (uint32_t col = 0; col < FACE_WIDTH; ++col) {
                row_ptr[col] = fill_value;
            }
        }
    }
}

/**
 * @brief Fills padding regions in a tile with a specified value.
 *
 * This function processes a tile by dividing it into faces and applying padding to each face
 * based on the unpadded dimensions. The padding is applied to both width and height dimensions
 * as needed.
 *
 * The function uses template metaprogramming to unroll the face processing loops at compile time,
 * making it efficient for hardware execution. It processes each face in the tile by:
 * 1. Calculating the face offset in the tile
 * 2. Determining the padding requirements for width and height
 * 3. Applying the padding with the specified fill value
 *
 * @tparam T The type of the elements in the tile
 * @tparam num_elements_unpadded_w Number of unpadded elements in the width dimension
 * @tparam num_elements_unpadded_h Number of unpadded elements in the height dimension
 * @tparam num_faces_w Number of faces in the width dimension (default: TILE_WIDTH / FACE_WIDTH)
 * @tparam num_faces_h Number of faces in the height dimension (default: TILE_HEIGHT / FACE_HEIGHT)
 * @param l1_tile_ptr Pointer to the start of the tile in L1 memory
 * @param fill_value The value to use for padding
 */
template <
    typename T,
    uint32_t num_elements_unpadded_w,
    uint32_t num_elements_unpadded_h,
    uint32_t num_faces_w = tt::constants::TILE_WIDTH / tt::constants::FACE_WIDTH,
    uint32_t num_faces_h = tt::constants::TILE_HEIGHT / tt::constants::FACE_HEIGHT>
void fill_pad_tile(uint32_t l1_tile_ptr, T fill_value) {
    auto tile_ptr = reinterpret_cast<T*>(l1_tile_ptr);

    // Face 0, 0
    fill_pad_face<T, num_elements_unpadded_w, num_elements_unpadded_h, num_faces_w, num_faces_h, 0, 0>(
        tile_ptr, fill_value);

    // Face 0, 1
    if constexpr (num_faces_w > 1) {
        fill_pad_face<T, num_elements_unpadded_w, num_elements_unpadded_h, num_faces_w, num_faces_h, 0, 1>(
            tile_ptr, fill_value);
    }

    // Face 1, 0
    if constexpr (num_faces_h > 1) {
        fill_pad_face<T, num_elements_unpadded_w, num_elements_unpadded_h, num_faces_w, num_faces_h, 1, 0>(
            tile_ptr, fill_value);
    }

    // Face 1, 1
    if constexpr (num_faces_w > 1 && num_faces_h > 1) {
        fill_pad_face<T, num_elements_unpadded_w, num_elements_unpadded_h, num_faces_w, num_faces_h, 1, 1>(
            tile_ptr, fill_value);
    }
}

// Turns a mantissa width into elements-per-byte. Unrelated to Bfp8's 8-bit mantissa.
constexpr uint32_t BITS_PER_BYTE = 8;

/**
 * @brief Zeroes a single element of a block-float mantissa row.
 *
 * Elements pack little-endian within the row's byte stream (element 0 in the LSBs, see
 * create_packed_bfp_packed_as_u32), so `col` occupies bits [bits * col, bits * (col + 1)).
 *
 * @tparam bits_per_element Mantissa width: 8 (Bfp8), 4 (Bfp4) or 2 (Bfp2)
 * @param face_row_ptr Pointer to the first mantissa byte of the face row
 * @param col The column within the face to zero
 */
template <uint32_t bits_per_element>
inline void zero_blockfloat_mantissa(uint8_t* face_row_ptr, uint32_t col) {
    constexpr uint32_t elements_per_byte = BITS_PER_BYTE / bits_per_element;
    if constexpr (elements_per_byte == 1) {
        // Bfp8: the element owns the whole byte, so no neighbours to preserve.
        face_row_ptr[col] = 0;
    } else {
        constexpr uint8_t element_mask = static_cast<uint8_t>((1u << bits_per_element) - 1);
        const uint32_t shift = (col % elements_per_byte) * bits_per_element;
        face_row_ptr[col / elements_per_byte] &= static_cast<uint8_t>(~(element_mask << shift));
    }
}

/**
 * @brief Zeroes the padded region of one face of a block-float tile's mantissa section.
 *
 * Mirrors fill_pad_face's padding math exactly, but indexes into a mantissa stream whose
 * elements are bits_per_element wide instead of a T-typed array.
 *
 * @tparam bits_per_element Mantissa width: 8 (Bfp8), 4 (Bfp4) or 2 (Bfp2)
 * @tparam num_elements_unpadded_w Number of unpadded elements in the width dimension
 * @tparam num_elements_unpadded_h Number of unpadded elements in the height dimension
 * @tparam num_faces_w Number of faces in the width dimension
 * @tparam num_faces_h Number of faces in the height dimension
 * @tparam face_h The height index of the face
 * @tparam face_w The width index of the face
 * @param mantissa_ptr Pointer to the start of the tile's mantissa section
 */
template <
    uint32_t bits_per_element,
    uint32_t num_elements_unpadded_w,
    uint32_t num_elements_unpadded_h,
    uint32_t num_faces_w,
    uint32_t num_faces_h,
    uint32_t face_h,
    uint32_t face_w>
void fill_pad_face_blockfloat(uint8_t* mantissa_ptr) {
    using namespace tt::constants;

    constexpr uint32_t elements_per_byte = BITS_PER_BYTE / bits_per_element;
    constexpr uint32_t face_row_bytes = FACE_WIDTH / elements_per_byte;

    // Calculate face offset
    constexpr uint32_t face_offset_bytes = (face_h * num_faces_w + face_w) * (FACE_HW / elements_per_byte);
    auto face_ptr = mantissa_ptr + face_offset_bytes;

    // Right padding (width padding)
    constexpr uint32_t face_w_offset = face_w * FACE_WIDTH;
    constexpr uint32_t face_pad_w = (num_elements_unpadded_w <= face_w_offset) ? FACE_WIDTH
                                    : (num_elements_unpadded_w >= face_w_offset + FACE_WIDTH)
                                        ? 0
                                        : face_w_offset + FACE_WIDTH - num_elements_unpadded_w;

    if constexpr (face_pad_w > 0) {
        for (uint32_t row = 0; row < FACE_HEIGHT; ++row) {
            auto row_ptr = face_ptr + row * face_row_bytes;
#pragma GCC unroll 32
            for (uint32_t col = FACE_WIDTH - face_pad_w; col < FACE_WIDTH; ++col) {
                zero_blockfloat_mantissa<bits_per_element>(row_ptr, col);
            }
        }
    }

    // Bottom padding (height padding) - the entire row is padded, so clear its bytes directly
    constexpr uint32_t face_h_offset = face_h * FACE_HEIGHT;
    constexpr uint32_t face_pad_h = (num_elements_unpadded_h <= face_h_offset) ? FACE_HEIGHT
                                    : (num_elements_unpadded_h >= face_h_offset + FACE_HEIGHT)
                                        ? 0
                                        : face_h_offset + FACE_HEIGHT - num_elements_unpadded_h;

    if constexpr (face_pad_h > 0) {
        for (uint32_t row = FACE_HEIGHT - face_pad_h; row < FACE_HEIGHT; ++row) {
            auto row_ptr = face_ptr + row * face_row_bytes;
#pragma GCC unroll 32
            for (uint32_t byte_index = 0; byte_index < face_row_bytes; ++byte_index) {
                row_ptr[byte_index] = 0;
            }
        }
    }
}

/**
 * @brief Fills padding regions of a block-float (shared-exponent) tile with zeros.
 *
 * Block-float tiles are laid out as a per-face-row exponent section followed by the mantissa
 * section (see the packing-order comment in blockfloat_common.cpp):
 *
 *     [ FACE_HEIGHT exponents for face 0 ] ... [ FACE_HEIGHT exponents for face N-1 ]
 *     [ face 0 mantissas (row-major) ]     ... [ face N-1 mantissas (row-major) ]
 *
 * Only mantissas are touched: a zero mantissa decodes to 0.0 whatever exponent it shares (see the
 * man == 0 branches in blockfloat_common.cpp), so elements sharing that exponent are unaffected.
 *
 * @tparam bits_per_element Mantissa width: 8 (Bfp8), 4 (Bfp4) or 2 (Bfp2)
 * @tparam num_elements_unpadded_w Number of unpadded elements in the width dimension
 * @tparam num_elements_unpadded_h Number of unpadded elements in the height dimension
 * @tparam num_faces_w Number of faces in the width dimension (default: TILE_WIDTH / FACE_WIDTH)
 * @tparam num_faces_h Number of faces in the height dimension (default: TILE_HEIGHT / FACE_HEIGHT)
 * @param l1_tile_ptr Pointer to the start of the tile in L1 memory
 */
template <
    uint32_t bits_per_element,
    uint32_t num_elements_unpadded_w,
    uint32_t num_elements_unpadded_h,
    uint32_t num_faces_w = tt::constants::TILE_WIDTH / tt::constants::FACE_WIDTH,
    uint32_t num_faces_h = tt::constants::TILE_HEIGHT / tt::constants::FACE_HEIGHT>
void fill_pad_tile_blockfloat(uint32_t l1_tile_ptr) {
    using namespace tt::constants;

    constexpr uint32_t exponent_section_bytes = num_faces_w * num_faces_h * FACE_HEIGHT;
    auto mantissa_ptr = reinterpret_cast<uint8_t*>(l1_tile_ptr) + exponent_section_bytes;

    // Face 0, 0
    fill_pad_face_blockfloat<
        bits_per_element,
        num_elements_unpadded_w,
        num_elements_unpadded_h,
        num_faces_w,
        num_faces_h,
        0,
        0>(mantissa_ptr);

    // Face 0, 1
    if constexpr (num_faces_w > 1) {
        fill_pad_face_blockfloat<
            bits_per_element,
            num_elements_unpadded_w,
            num_elements_unpadded_h,
            num_faces_w,
            num_faces_h,
            0,
            1>(mantissa_ptr);
    }

    // Face 1, 0
    if constexpr (num_faces_h > 1) {
        fill_pad_face_blockfloat<
            bits_per_element,
            num_elements_unpadded_w,
            num_elements_unpadded_h,
            num_faces_w,
            num_faces_h,
            1,
            0>(mantissa_ptr);
    }

    // Face 1, 1
    if constexpr (num_faces_w > 1 && num_faces_h > 1) {
        fill_pad_face_blockfloat<
            bits_per_element,
            num_elements_unpadded_w,
            num_elements_unpadded_h,
            num_faces_w,
            num_faces_h,
            1,
            1>(mantissa_ptr);
    }
}

/**
 * @brief Mantissa bits per element for a block-float data format, or 0 if the format is not a
 *        block-float format.
 */
template <DataFormat data_format>
constexpr uint32_t blockfloat_mantissa_bits() {
#ifndef ARCH_QUASAR
    if constexpr (data_format == DataFormat::Bfp8 || data_format == DataFormat::Bfp8_b) {
        return 8;
    } else if constexpr (data_format == DataFormat::Bfp4 || data_format == DataFormat::Bfp4_b) {
        return 4;
    } else if constexpr (data_format == DataFormat::Bfp2 || data_format == DataFormat::Bfp2_b) {
        return 2;
    } else {
        return 0;
    }
#else
    // Quasar's DataFormat enum has NO WH-style block-float formats: the codes WH/BH use for Bfp8/Bfp4/Bfp2
    // (2/3/11) are reused on Quasar for the microscaling MxInt8/MxInt4/MxInt2, and there is no Bfp8_b/Bfp4_b/
    // Bfp2_b at all. So this WH-Bfp mantissa accounting does not apply on Quasar; report 0 (same as the
    // non-block-float branch on WH/BH). The one block-float DataType a caller could otherwise pass, BFLOAT8_B,
    // is rejected up front by the quasar matmul op (experimental/quasar/matmul/matmul.cpp) -- the only Quasar
    // caller of pad_last_ktile* -- so a block-float format never reaches here on Quasar (no silent mis-pad).
    // A value-based assert is deliberately NOT added: the codes 2/3/11 are legitimate Quasar MxInt formats,
    // so asserting on them would false-fire; the op-level dtype reject is the correct guard. (Padding for
    // Quasar's Mx microscaling formats, if ever needed, is a separate Mx-specific path -- not this.)
    return 0;
#endif
}

/**
 * @brief Pads the last K tile in a matrix multiplication operation.
 *
 * This function handles padding for the last K tile in a matrix multiplication operation.
 * It applies zero padding based on the specified data format (Float32, Float16_b, or any of the
 * block-float formats) and the unpadded width of the last K tile.
 *
 * @tparam in0_data_format The data format of the input tensor
 * @tparam in0_last_ktile_w The unpadded width of the last K tile
 * @param l1_write_addr_in0 The L1 memory address where the zeros should be written
 */
template <DataFormat in0_data_format, uint32_t in0_last_ktile_w>
void pad_last_ktile(uint32_t l1_write_addr_in0) {
    using namespace tt::constants;
    // Non-zero width means block-float; blockfloat_mantissa_bits() returns 0 for everything else.
    constexpr uint32_t mantissa_bits = blockfloat_mantissa_bits<in0_data_format>();
    constexpr bool is_blockfloat = mantissa_bits > 0;
    if constexpr (in0_data_format == DataFormat::Float32) {
        fill_pad_tile<uint32_t, in0_last_ktile_w, /*num_elements_unpadded_h=*/TILE_HEIGHT>(
            l1_write_addr_in0, /*pad_value=*/0);
    } else if constexpr (in0_data_format == DataFormat::Float16_b) {
        fill_pad_tile<uint16_t, in0_last_ktile_w, /*num_elements_unpadded_h=*/TILE_HEIGHT>(
            l1_write_addr_in0, /*pad_value=*/0);
    } else if constexpr (is_blockfloat) {
        fill_pad_tile_blockfloat<mantissa_bits, in0_last_ktile_w, /*num_elements_unpadded_h=*/TILE_HEIGHT>(
            l1_write_addr_in0);
    }
}

/**
 * @brief Pads the last K tile when the input is transposed.
 *
 * When transpose_a is true, K maps to the height dimension of the physical tile.
 * This function applies zero padding to the height (rows) of the last K tile,
 * leaving width fully unpadded.
 *
 * @tparam in0_data_format The data format of the input tensor
 * @tparam in0_last_ktile_h The unpadded height of the last K tile
 * @param l1_write_addr_in0 The L1 memory address where the zeros should be written
 */
template <DataFormat in0_data_format, uint32_t in0_last_ktile_h>
void pad_last_transposed_ktile(uint32_t l1_write_addr_in0) {
    using namespace tt::constants;
    // Non-zero width means block-float; blockfloat_mantissa_bits() returns 0 for everything else.
    constexpr uint32_t mantissa_bits = blockfloat_mantissa_bits<in0_data_format>();
    constexpr bool is_blockfloat = mantissa_bits > 0;
    if constexpr (in0_data_format == DataFormat::Float32) {
        fill_pad_tile<uint32_t, /*num_elements_unpadded_w=*/TILE_WIDTH, in0_last_ktile_h>(
            l1_write_addr_in0, /*pad_value=*/0);
    } else if constexpr (in0_data_format == DataFormat::Float16_b) {
        fill_pad_tile<uint16_t, /*num_elements_unpadded_w=*/TILE_WIDTH, in0_last_ktile_h>(
            l1_write_addr_in0, /*pad_value=*/0);
    } else if constexpr (is_blockfloat) {
        fill_pad_tile_blockfloat<mantissa_bits, /*num_elements_unpadded_w=*/TILE_WIDTH, in0_last_ktile_h>(
            l1_write_addr_in0);
    }
}
