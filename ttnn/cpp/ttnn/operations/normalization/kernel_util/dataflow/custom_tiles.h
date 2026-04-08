// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"

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
FORCE_INLINE void generate_partial_reduce_scaler(
    const uint32_t cb_id,
    const uint32_t scaler,
    const uint32_t num_cols,
    const uint32_t tile_height = 32,
    const uint32_t tile_width = 32) {
    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id);

    cb.reserve_back(1);

    const uint16_t scaler_uint16 = scaler >> 16;

    constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
    static_assert(num_zeros_reads > 0, "num_zeros_reads must be greater than 0");
    uint32_t write_addr = cb.get_write_ptr();
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(write_addr);

    // Fill tile with zeros via state-based reads from the local L1 zeros region
    experimental::UnicastEndpoint self;
    noc.set_async_read_state<experimental::Noc::VcSelection::DEFAULT, NOC_MAX_BURST_SIZE>(
        self, MEM_ZEROS_SIZE, {.noc_x = my_x[0], .noc_y = my_y[0], .addr = MEM_ZEROS_BASE});
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc.async_read_with_state<experimental::Noc::VcSelection::DEFAULT, NOC_MAX_BURST_SIZE>(
            self,
            experimental::CoreLocalMem<uint32_t>(write_addr),
            MEM_ZEROS_SIZE,
            {.noc_x = my_x[0], .noc_y = my_y[0], .addr = MEM_ZEROS_BASE},
            {});
        write_addr += MEM_ZEROS_SIZE;
    }
    noc.async_read_barrier();

    // Iterate over first two faces (top two) then
    // second two faces (bottom two)
    if (scaler_uint16 != 0) {
        const uint32_t face_rows = tile_height / tt::constants::FACE_HEIGHT;
        const uint32_t faces_per_row = tile_width / tt::constants::FACE_WIDTH;
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

    cb.push_back(1);
}

}  // namespace norm::kernel_util::dataflow
