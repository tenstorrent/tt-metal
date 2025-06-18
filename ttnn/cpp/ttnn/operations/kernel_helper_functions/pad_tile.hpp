// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/constants.hpp>

using namespace tt::constants;

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
#pragma unroll
        for (uint32_t row = 0; row < FACE_HEIGHT; ++row) {
            auto row_ptr = face_ptr + row * FACE_WIDTH;
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
#pragma unroll
        for (uint32_t row = FACE_HEIGHT - face_pad_h; row < FACE_HEIGHT; ++row) {
            auto row_ptr = face_ptr + row * FACE_WIDTH;
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
    uint32_t num_faces_w = TILE_WIDTH / FACE_WIDTH,
    uint32_t num_faces_h = TILE_HEIGHT / FACE_HEIGHT>
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

/**
 * @brief Pads the last K tile in a matrix multiplication operation.
 *
 * This function handles padding for the last K tile in a matrix multiplication operation.
 * It applies zero padding based on the specified data format (Float32 or Float16_b) and
 * the unpadded width of the last K tile.
 *
 * @tparam in0_data_format The data format of the input tensor (Float32 or Float16_b)
 * @tparam in0_last_ktile_w The unpadded width of the last K tile
 * @param l1_write_addr_in0 The L1 memory address where the zeros should be written
 */
template <DataFormat in0_data_format, uint32_t in0_last_ktile_w>
void pad_last_ktile(uint32_t l1_write_addr_in0) {
    if constexpr (in0_data_format == DataFormat::Float32) {
        fill_pad_tile<uint32_t, in0_last_ktile_w, /*num_elements_unpadded_h=*/TILE_HEIGHT>(
            l1_write_addr_in0, /*pad_value=*/0);
    } else if constexpr (in0_data_format == DataFormat::Float16_b) {
        fill_pad_tile<uint16_t, in0_last_ktile_w, /*num_elements_unpadded_h=*/TILE_HEIGHT>(
            l1_write_addr_in0, /*pad_value=*/0);
    }
}
