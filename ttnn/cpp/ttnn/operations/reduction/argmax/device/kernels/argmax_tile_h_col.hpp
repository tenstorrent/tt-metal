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
 *
 * Loop order: col outer, row inner — keeps the per-column accumulator in a register
 * for the full row scan, matching the width kernel's register-caching pattern.
 * Bounds checks are handled at face granularity via get_face_data_range, so no
 * per-element global_h/global_w checks are needed.
 */
template <typename DTYPE, DataFormat format>
void process_loaded_tile_all_h_columns(
    const InputContext& ctx, uint32_t w_tile, uint32_t h_tile, DTYPE max_vals[], uint32_t arg_maxs[]) {
    auto src_ptr = get_tt_l1_ptr_based_on_data_format<format>(ctx.cb_addr);

    constexpr uint32_t faces_per_tile = 4;
    for (uint32_t face_id = 0; face_id < faces_per_tile; face_id++) {
        uint32_t rows_to_process = face_height;
        uint32_t cols_to_process = face_width;
        if (ctx.has_padding) {
            get_face_data_range(rows_to_process, cols_to_process, w_tile, h_tile, face_id, ctx);
            ASSERT(rows_to_process <= face_height);
            ASSERT(cols_to_process <= face_width);
        }

        if (rows_to_process == 0 || cols_to_process == 0) {
            continue;
        }

        const uint32_t face_offset = face_id * face_size;
        volatile tt_l1_ptr DTYPE* face_ptr = src_ptr + face_offset;

        // Hoist face-level invariants out of the element loops.
        const bool is_left_face = (face_id == 0 || face_id == 2);
        const uint32_t col_offset = is_left_face ? 0 : face_width;
        // Base row-within-tile for this face (bottom faces start at face_height).
        const uint32_t base_row_in_tile = (face_id < 2) ? 0 : face_height;
        const uint32_t base_global_h = h_tile * ctx.tile_height + base_row_in_tile;

        // Outer loop: column — load accumulator once into a register, scan all rows,
        // store back once.  This matches the register-caching pattern of the width
        // kernel and avoids a L1 load+store on every element comparison.
        for (uint32_t col = 0; col < cols_to_process; col++) {
            const uint32_t in_tile_w = col_offset + col;
            DTYPE curr_max = max_vals[in_tile_w];
            uint32_t curr_arg = arg_maxs[in_tile_w];

            // Inner loop: row — stride through this column across all rows.
            volatile tt_l1_ptr DTYPE* col_ptr = face_ptr + col;
            for (uint32_t row = 0; row < rows_to_process; row++) {
                const DTYPE value = col_ptr[row * face_width];
                bool new_max = false;
                if constexpr (format == DataFormat::Float16_b) {
                    new_max = bfloat16_greater(value, curr_max);
                } else if constexpr (format == DataFormat::Float32) {
                    new_max = float32_greater(value, curr_max);
                }
                if (new_max) {
                    curr_max = value;
                    curr_arg = base_global_h + row;
                }
            }
            max_vals[in_tile_w] = curr_max;
            arg_maxs[in_tile_w] = curr_arg;
        }
    }
}
