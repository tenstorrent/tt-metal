// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Moving one column of a tile into column 0 of another, which is where a column broadcast
// reads its operand and where the read takes a per-token scalar from.
//
// A tile is four 16x16 faces, row-major within a face and face-major across the tile, so
// column `col` is sixteen elements at a stride of 16 in each of two faces: 0 and 2 for the
// left half of the tile, 1 and 3 for the right.

#pragma once

#include <cstdint>

template <typename T>
inline void copy_column_to_column0(uint32_t src_addr, uint32_t dst_addr, uint32_t col) {
    constexpr uint32_t kFaceDim = 16;
    constexpr uint32_t kFaceElems = kFaceDim * kFaceDim;
    const auto* src =
        reinterpret_cast<const volatile tt_l1_ptr T*>(src_addr) + (col / kFaceDim) * kFaceElems + col % kFaceDim;
    auto* dst = reinterpret_cast<volatile tt_l1_ptr T*>(dst_addr);
    for (uint32_t face : {0u, 2u}) {
        for (uint32_t row = 0; row < kFaceDim; ++row) {
            const uint32_t offset = face * kFaceElems + row * kFaceDim;
            dst[offset] = src[offset];
        }
    }
}
