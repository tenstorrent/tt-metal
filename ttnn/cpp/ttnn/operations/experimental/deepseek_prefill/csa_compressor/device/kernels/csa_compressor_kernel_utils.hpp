// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>

#include "api/dataflow/dataflow_api.h"

namespace csa_compressor::kernel_utils {

constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kTileElems = 1024;

inline uint32_t tile_offset(uint32_t row, uint32_t col) {
    const uint32_t face = (row / 16) * 2 + col / 16;
    return face * 256 + (row % 16) * 16 + col % 16;
}

inline float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline uint16_t float_to_bf16_rne(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

// One tile row is two runs of 16 contiguous elements, one per face column, and both start on a
// 32-byte boundary. Moving them as words keeps the copy to 16 stores per row.
inline void copy_tile_row(
    volatile tt_l1_ptr uint16_t* destination_tile,
    volatile tt_l1_ptr uint16_t* source_tile,
    uint32_t destination_row,
    uint32_t source_row) {
    for (uint32_t face_col = 0; face_col < 32; face_col += 16) {
        volatile tt_l1_ptr uint32_t* destination =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_tile + tile_offset(destination_row, face_col));
        volatile tt_l1_ptr uint32_t* source =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source_tile + tile_offset(source_row, face_col));
        for (uint32_t word = 0; word < 8; ++word) {
            destination[word] = source[word];
        }
    }
}

}  // namespace csa_compressor::kernel_utils
