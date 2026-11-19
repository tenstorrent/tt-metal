// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Called after the read barrier. Padding is not part of the tensor's value:
// clear it before matmul/quantization, including NaN/Inf and BFP exponents.
template <uint32_t TileBytes>
FORCE_INLINE void zero_tile_padding(uint32_t address, uint32_t valid_rows) {
    static_assert(TileBytes == 2048 || TileBytes == 1088 || TileBytes == 576);
    constexpr uint32_t header = TileBytes == 2048 ? 0 : 64;
    constexpr uint32_t row_words = (TileBytes - header) / 256;
    auto* values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address + header);
    auto* exponents = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(address);
    for (uint32_t row = valid_rows; row < 32; ++row) {
        for (uint32_t col_face = 0; col_face < 2; ++col_face) {
            const uint32_t face_row = (row / 16 * 2 + col_face) * 16 + row % 16;
            for (uint32_t word = 0; word < row_words; ++word) {
                values[face_row * row_words + word] = 0;
            }
            if constexpr (header != 0) {
                exponents[face_row] = 0;
            }
        }
    }
}
