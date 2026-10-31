// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#ifndef ALWI
#define ALWI inline __attribute__((always_inline))
#endif

namespace ttnn::operations::wavelet::kernels::primitives {

constexpr uint32_t kTileSide = 32;
constexpr uint32_t kFaceSide = 16;
constexpr uint32_t kFaceElements = kFaceSide * kFaceSide;
constexpr uint32_t kTileElements = kTileSide * kTileSide;
constexpr uint32_t kTileBytes = kTileElements * sizeof(float);

[[nodiscard]] constexpr uint32_t tile_face_row_offset(const uint32_t row) {
    return (row / kFaceSide) * 2 * kFaceElements + (row % kFaceSide) * kFaceSide;
}

[[nodiscard]] constexpr uint32_t tile_face_column_offset(const uint32_t column) {
    return (column / kFaceSide) * kFaceElements + column % kFaceSide;
}

[[nodiscard]] constexpr uint32_t tile_element_offset(const uint32_t row, const uint32_t column) {
    return tile_face_row_offset(row) + tile_face_column_offset(column);
}

[[nodiscard]] constexpr uint32_t tiled_element_offset(
    const uint32_t row, const uint32_t column, const uint32_t tile_columns) {
    const uint32_t tile_index = (row / kTileSide) * tile_columns + column / kTileSide;
    return tile_index * kTileElements + tile_element_offset(row % kTileSide, column % kTileSide);
}

static_assert(tile_element_offset(0, 0) == 0);
static_assert(tile_element_offset(0, 16) == kFaceElements);
static_assert(tile_element_offset(16, 0) == 2 * kFaceElements);
static_assert(tile_element_offset(31, 31) == kTileElements - 1);

}  // namespace ttnn::operations::wavelet::kernels::primitives
