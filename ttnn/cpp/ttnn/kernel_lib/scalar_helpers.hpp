// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace dataflow_kernel_lib {

// Face dimensions (always 16x16 elements regardless of element size)
constexpr uint32_t FACE_ROWS = 16;
constexpr uint32_t FACE_COLS = 16;
constexpr uint32_t FACE_ELEMENTS = FACE_ROWS * FACE_COLS;  // 256 elements per face

// ============================================================================
// 16-bit element functions (bfloat16)
// Scalar is a 16-bit value double packed into a u32: (bf16 << 16 | bf16)
// ============================================================================

/**
 * @brief Generate a column broadcast (W-bcast) scalar tile for 16-bit elements
 *
 * Creates a tile with the scalar value placed in column 0 of faces 0 and 2
 * (left column of faces). Used for width broadcast operations.
 *
 * @param cb_id Circular buffer ID to write the tile to
 * @param scaler Packed bf16 value (bf16 << 16 | bf16)
 */
FORCE_INLINE void generate_bcast_col_scalar_bfloat16(const uint32_t cb_id, const uint32_t scaler);

/**
 * @brief Generate a row broadcast (H-bcast) scalar tile for 16-bit elements
 *
 * Creates a tile with the scalar value placed in row 0 of faces 0 and 1
 * (top row of faces). Used for height broadcast operations.
 *
 * @param cb_id Circular buffer ID to write the tile to
 * @param scaler Packed bf16 value (bf16 << 16 | bf16)
 */
FORCE_INLINE void generate_bcast_row_scalar_bfloat16(const uint32_t cb_id, const uint32_t scaler);

/**
 * @brief Generate a full broadcast (HW-bcast) scalar tile for 16-bit elements
 *
 * Creates a tile with a single scalar value at position 0 of face 0.
 * Used for scalar operations where the value broadcasts across both dimensions.
 *
 * @param cb_id Circular buffer ID to write the tile to
 * @param scaler Packed bf16 value (bf16 << 16 | bf16)
 */
FORCE_INLINE void generate_bcast_scalar_bfloat16(const uint32_t cb_id, const uint32_t scaler);

// ============================================================================
// 32-bit element functions (float32, int32)
// Scalar is a single 32-bit value
// ============================================================================

/**
 * @brief Generate a column broadcast (W-bcast) scalar tile for 32-bit elements
 *
 * Creates a tile with the scalar value placed in column 0 of faces 0 and 2
 * (left column of faces). Used for width broadcast operations.
 *
 * @param cb_id Circular buffer ID to write the tile to
 * @param scaler 32-bit scalar value (float32 or int32)
 */
FORCE_INLINE void generate_bcast_col_scalar(const uint32_t cb_id, const uint32_t scaler);

/**
 * @brief Generate a row broadcast (H-bcast) scalar tile for 32-bit elements
 *
 * Creates a tile with the scalar value placed in row 0 of faces 0 and 1
 * (top row of faces). Used for height broadcast operations.
 *
 * @param cb_id Circular buffer ID to write the tile to
 * @param scaler 32-bit scalar value (float32 or int32)
 */
FORCE_INLINE void generate_bcast_row_scalar(const uint32_t cb_id, const uint32_t scaler);

/**
 * @brief Generate a full broadcast (HW-bcast) scalar tile for 32-bit elements
 *
 * Creates a tile with a single scalar value at position 0 of face 0.
 * Used for scalar operations where the value broadcasts across both dimensions.
 *
 * @param cb_id Circular buffer ID to write the tile to
 * @param scaler 32-bit scalar value (float32 or int32)
 */
FORCE_INLINE void generate_bcast_scalar(const uint32_t cb_id, const uint32_t scaler);

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/scalar_helpers.inl"
