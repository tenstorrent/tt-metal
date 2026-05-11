// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "argmax_common.hpp"
#include "argmax_tile_layout.hpp"
#include "api/debug/assert.h"

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

/**
 * One pass over the tile already in the CB: for each in-tile column `in_tile_w`, update
 * max_vals[in_tile_w] / arg_maxs[in_tile_w] (argmax along global H for that global_w column).
 * Matches face indexing and padding handling of process_input_tile (W reader).
 */
template <typename DTYPE, DataFormat format>
void process_loaded_tile_all_h_columns(
    const InputContext& ctx, uint32_t w_tile, uint32_t h_tile, DTYPE max_vals[], uint32_t arg_maxs[]) {
    const bool has_padding = ctx.has_padding;
    (void)has_padding;
    auto src_ptr = get_tt_l1_ptr_based_on_data_format<format>(ctx.cb_addr);

    for (uint32_t face_id = 0; face_id < 4; face_id++) {
        uint32_t rows_to_process = face_width;
        uint32_t cols_to_process = face_height;
        if (ctx.has_padding) {
            get_face_data_range(rows_to_process, cols_to_process, w_tile, h_tile, face_id, ctx);
            ASSERT(rows_to_process <= face_height);
            ASSERT(cols_to_process <= face_width);
        }

        if (rows_to_process == 0 && cols_to_process == 0) {
            continue;
        }

        const uint32_t face_offset = face_id * face_size;
        volatile tt_l1_ptr DTYPE* face_ptr = src_ptr + face_offset;

        for (uint32_t row = 0; row < rows_to_process; row++) {
            const uint32_t row_in_tile = (face_id < 2) ? row : row + face_height;

            for (uint32_t col = 0; col < cols_to_process; col++) {
                const bool is_left_side_face = (face_id == 0 || face_id == 2);
                const uint32_t in_tile_w = (is_left_side_face ? 0U : face_width) + col;

                const uint32_t index = row * face_width + col;
                const DTYPE value = face_ptr[index];
                const uint32_t global_h = h_tile * ctx.tile_height + row_in_tile;
                if (global_h >= ctx.logical_height) {
                    continue;
                }

                const uint32_t global_w = w_tile * ctx.tile_width + in_tile_w;
                if (global_w >= ctx.logical_width) {
                    continue;
                }

                DTYPE& max_val = max_vals[in_tile_w];
                uint32_t& arg_max = arg_maxs[in_tile_w];

                bool new_max = false;
                if constexpr (format == DataFormat::Float16_b) {
                    new_max = bfloat16_greater(value, max_val);
                } else if constexpr (format == DataFormat::Float32) {
                    new_max = float32_greater(value, max_val);
                }
                if (new_max) {
                    max_val = value;
                    arg_max = global_h;
                }
            }
        }
    }
}
