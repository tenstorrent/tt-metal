// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file custom_tiles.h
 * @brief Functions to generate various special-purpose tiles in
 * dataflow kernels
 *
 */

#pragma once

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/constants.hpp>

namespace norm::kernel_util::dataflow {

/**
 * @brief Generate a partial reduce scaler tile for reducing
 * only the first `num_cols` columns of a tile
 * @param cb_id The ID of the CB to generate the tile for
 * @param scaler The scaler value to generate the tile for.
 * Should be two 16-bit values double packed into a uint32_t
 * @param num_cols The number of columns in the tile that will
 * participate in the reduction
 */
FORCE_INLINE void generate_partial_reduce_scaler(const uint32_t cb_id, const uint32_t scaler, const uint32_t num_cols) {
    cb_reserve_back(cb_id, 1);

    const uint16_t scaler_uint16 = scaler >> 16;

    constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
    static_assert(num_zeros_reads > 0, "num_zeros_reads must be greater than 0");
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(write_addr);

    // Fill tile with zeros
    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();

    // Iterate over first two faces (top two) then
    // second two faces (bottom two)
    if (scaler_uint16 != 0) {
        constexpr uint32_t face_rows = tt::constants::TILE_HEIGHT / tt::constants::FACE_HEIGHT;
        constexpr uint32_t faces_per_row = tt::constants::TILE_WIDTH / tt::constants::FACE_WIDTH;
        constexpr uint32_t cols_per_face = tt::constants::FACE_WIDTH;
        uint32_t face_row_offset = 0;
        for (uint32_t i = 0; i < face_rows; ++i) {
            uint32_t face_offset_in_row = 0;
            for (uint32_t j = 0; j < faces_per_row; ++j) {
                for (uint32_t k = 0; k < cols_per_face; ++k) {
                    uint32_t col = j * cols_per_face + k;
                    if (col < num_cols) {
                        ptr[face_row_offset + face_offset_in_row + k] = scaler_uint16;
                    }
                }
                face_offset_in_row += tt::constants::FACE_HW;
            }
            face_row_offset += faces_per_row * tt::constants::FACE_HW;
        }
    }

    cb_push_back(cb_id, 1);
}

}  // namespace norm::kernel_util::dataflow
