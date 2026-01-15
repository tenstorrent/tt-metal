// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

namespace ttnn::kernel_lib::dataflow {

// Face size in uint32 (128 u32 = 256 bf16 = 16x16 face)
constexpr uint32_t FACE_SIZE_U32 = 128;
// Row size in uint32 (8 u32 = 16 bf16)
constexpr uint32_t ROW_SIZE_U32 = 8;

namespace detail {

template <bool half_tile>
FORCE_INLINE void zero_faces(uint32_t write_addr) {
    constexpr uint32_t num_faces = half_tile ? 2 : 4;
    constexpr uint32_t bytes_to_zero = num_faces * FACE_SIZE_U32 * sizeof(uint32_t);
    constexpr uint32_t num_zeros_reads = bytes_to_zero / MEM_ZEROS_SIZE;

    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);

    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();
}

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

}  // namespace detail

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
    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    detail::zero_faces<half_tile>(write_addr);

    if (scaler != 0) {
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
        detail::fill_row0<half_tile>(ptr, scaler);
    }

    cb_push_back(cb_id, 1);
}

}  // namespace ttnn::kernel_lib::dataflow
