// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/constants.hpp>
#include "api/numeric/bfloat16.h"
#include "api/numeric/float32.h"

/**
 * @brief Enumeration for neutral value policies used in reduction operations.
 *
 * Different reduction operations require different neutral (identity) values
 * for padding to ensure correct results:
 * - Zero: For sum/mean operations (adding 0 doesn't change the sum)
 * - NegInf: For max operations (max with -inf gives the original value)
 * - PosInf: For min operations (min with +inf gives the original value)
 */
enum class NeutralPolicy : uint32_t {
    Zero = 0,    // Neutral element for sum/mean
    NegInf = 1,  // Neutral element for max
    PosInf = 2,  // Neutral element for min
};

/**
 * @brief Get the neutral fill value for a given data format and policy.
 *
 * @tparam data_format The data format (Float32, Bfloat16, etc.)
 * @tparam policy The neutral value policy
 * @return The appropriate fill value as a uint32_t (for Float32) or uint16_t (for Bfloat16)
 */
template <tt::DataFormat data_format, NeutralPolicy policy>
constexpr auto get_neutral_value() {
    if constexpr (data_format == tt::DataFormat::Float32) {
        if constexpr (policy == NeutralPolicy::Zero) {
            return static_cast<uint32_t>(0);
        } else if constexpr (policy == NeutralPolicy::NegInf) {
            return NEG_INF_FLOAT32;
        } else if constexpr (policy == NeutralPolicy::PosInf) {
            return POS_INF_FLOAT32;
        }
    } else {
        // Default to Bfloat16 for all other formats
        if constexpr (policy == NeutralPolicy::Zero) {
            return static_cast<uint16_t>(0);
        } else if constexpr (policy == NeutralPolicy::NegInf) {
            return NEG_INF_BFLOAT16;
        } else if constexpr (policy == NeutralPolicy::PosInf) {
            return POS_INF_BFLOAT16;
        }
    }
}

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
    using namespace tt::constants;
    if constexpr (in0_data_format == DataFormat::Float32) {
        fill_pad_tile<uint32_t, in0_last_ktile_w, /*num_elements_unpadded_h=*/TILE_HEIGHT>(
            l1_write_addr_in0, /*pad_value=*/0);
    } else if constexpr (in0_data_format == DataFormat::Float16_b) {
        fill_pad_tile<uint16_t, in0_last_ktile_w, /*num_elements_unpadded_h=*/TILE_HEIGHT>(
            l1_write_addr_in0, /*pad_value=*/0);
    }
}

/**
 * @brief Pads the last width tile in a reduction operation.
 *
 * This function handles padding for the last tile in the width dimension of a reduction operation.
 * It applies the appropriate neutral value based on the data format and neutral policy.
 *
 * @tparam data_format The data format of the input tensor (Float32 or Bfloat16)
 * @tparam last_tile_w The unpadded width of the last tile (1 to TILE_WIDTH)
 * @tparam policy The neutral value policy (Zero for sum/mean, NegInf for max, PosInf for min)
 * @param l1_write_addr The L1 memory address where the padding should be written
 */
template <tt::DataFormat data_format, uint32_t last_tile_w, NeutralPolicy policy>
void pad_last_wtile(uint32_t l1_write_addr) {
    using namespace tt::constants;
    if constexpr (data_format == tt::DataFormat::Float32) {
        constexpr uint32_t fill_value = get_neutral_value<data_format, policy>();
        fill_pad_tile<uint32_t, last_tile_w, /*num_elements_unpadded_h=*/TILE_HEIGHT>(l1_write_addr, fill_value);
    } else {
        // Default to Bfloat16 for all other formats
        constexpr uint16_t fill_value = get_neutral_value<data_format, policy>();
        fill_pad_tile<uint16_t, last_tile_w, /*num_elements_unpadded_h=*/TILE_HEIGHT>(l1_write_addr, fill_value);
    }
}

/**
 * @brief Pads the last height tile in a reduction operation.
 *
 * This function handles padding for the last tile in the height dimension of a reduction operation.
 * It applies the appropriate neutral value based on the data format and neutral policy.
 *
 * @tparam data_format The data format of the input tensor (Float32 or Bfloat16)
 * @tparam last_tile_h The unpadded height of the last tile (1 to TILE_HEIGHT)
 * @tparam policy The neutral value policy (Zero for sum/mean, NegInf for max, PosInf for min)
 * @param l1_write_addr The L1 memory address where the padding should be written
 */
template <tt::DataFormat data_format, uint32_t last_tile_h, NeutralPolicy policy>
void pad_last_htile(uint32_t l1_write_addr) {
    using namespace tt::constants;
    if constexpr (data_format == tt::DataFormat::Float32) {
        constexpr uint32_t fill_value = get_neutral_value<data_format, policy>();
        fill_pad_tile<uint32_t, /*num_elements_unpadded_w=*/TILE_WIDTH, last_tile_h>(l1_write_addr, fill_value);
    } else {
        // Default to Bfloat16 for all other formats
        constexpr uint16_t fill_value = get_neutral_value<data_format, policy>();
        fill_pad_tile<uint16_t, /*num_elements_unpadded_w=*/TILE_WIDTH, last_tile_h>(l1_write_addr, fill_value);
    }
}

/**
 * @brief Applies width padding based on data format, dispatching to the appropriate typed function.
 *
 * This is a helper function that eliminates repeated if-constexpr blocks in reader kernels.
 * It dispatches to pad_last_wtile with the correct data format type.
 *
 * @tparam IN_DF The input data format as uint32_t (cast of tt::DataFormat)
 * @tparam LAST_W The unpadded width of the last tile
 * @tparam policy The neutral value policy
 * @param l1_write_addr The L1 memory address where the padding should be written
 */
template <uint32_t IN_DF, uint32_t LAST_W, NeutralPolicy policy>
void apply_width_padding(uint32_t l1_write_addr) {
    if constexpr (IN_DF == static_cast<uint32_t>(tt::DataFormat::Bfloat16)) {
        pad_last_wtile<tt::DataFormat::Bfloat16, LAST_W, policy>(l1_write_addr);
    } else {
        pad_last_wtile<tt::DataFormat::Float32, LAST_W, policy>(l1_write_addr);
    }
}

/**
 * @brief Applies height padding based on data format, dispatching to the appropriate typed function.
 *
 * This is a helper function that eliminates repeated if-constexpr blocks in reader kernels.
 * It dispatches to pad_last_htile with the correct data format type.
 *
 * @tparam IN_DF The input data format as uint32_t (cast of tt::DataFormat)
 * @tparam LAST_H The unpadded height of the last tile
 * @tparam policy The neutral value policy
 * @param l1_write_addr The L1 memory address where the padding should be written
 */
template <uint32_t IN_DF, uint32_t LAST_H, NeutralPolicy policy>
void apply_height_padding(uint32_t l1_write_addr) {
    if constexpr (IN_DF == static_cast<uint32_t>(tt::DataFormat::Bfloat16)) {
        pad_last_htile<tt::DataFormat::Bfloat16, LAST_H, policy>(l1_write_addr);
    } else {
        pad_last_htile<tt::DataFormat::Float32, LAST_H, policy>(l1_write_addr);
    }
}
