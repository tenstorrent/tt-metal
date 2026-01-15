// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace dataflow_kernel_lib {

// Row size in uint32 (8 u32 = 16 bf16)
constexpr uint32_t ROW_SIZE_U32 = 8;

/**
 * @brief Fill row 0 of each face with a scaler value
 *
 * Writes the scaler value to columns 0-7 (the first 8 uint32_t positions)
 * of row 0 in each face. This corresponds to 16 bf16 values per face.
 *
 * @tparam half_tile If true, fill faces 0-1 only. If false, fill all 4 faces.
 * @param ptr Pointer to the start of the tile in L1 memory
 * @param scaler Packed bf16 value to write (bf16 << 16 | bf16)
 */
template <bool half_tile>
FORCE_INLINE void fill_row0(volatile tt_l1_ptr uint32_t* ptr, uint32_t scaler) {
    constexpr uint32_t num_faces = half_tile ? 2 : 4;

    for (uint32_t face = 0; face < num_faces; ++face) {
        uint32_t face_offset = face * FACE_SIZE_U32;
        for (uint32_t column = 0; column < ROW_SIZE_U32; ++column) {
            ptr[face_offset + column] = scaler;
        }
    }
}

/**
 * @brief Generate a reduce scaler tile
 *
 * Creates a tile in the specified circular buffer with the scaler value
 * placed in row 0 of each face. The tile is first zeroed, then
 * positions [0-7] of row 0 in each face are filled with the scaler.
 *
 * This is the standard implementation used for row-wise reduction operations.
 * The scaler is typically 1.0 for SUM/MAX reductions, and 1/N for AVG reductions.
 *
 * @tparam half_tile If true, only fill faces 0-1 (half tile mode)
 * @param cb_id Circular buffer ID to write the tile to
 * @param scaler Packed bf16 value (bf16 << 16 | bf16)
 *
 * @note The function handles cb_reserve_back and cb_push_back internally
 * @note If scaler is 0, the tile is left as all zeros
 */
template <bool half_tile = false>
FORCE_INLINE void generate_reduce_scaler(const uint32_t cb_id, const uint32_t scaler) {
    ASSERT(cb_id < NUM_CIRCULAR_BUFFERS);
    // Verify scaler is properly packed: high 16 bits must equal low 16 bits
    ASSERT((scaler >> 16) == (scaler & 0xFFFF));

    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    zero_faces<half_tile>(write_addr);

    if (scaler != 0) {
        fill_row0<half_tile>(addr_to_l1_ptr(write_addr), scaler);
    }

    cb_push_back(cb_id, 1);
}

}  // namespace dataflow_kernel_lib
