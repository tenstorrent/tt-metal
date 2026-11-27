// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Output-tile walk of regime N, shared by the reader and the writer so both classify every tile
// from the same compile-time geometry block (g6_pad_program_descriptor.geometry_ct_args).
//
// Output tile w = ((lead_0 * out_lead_1 + lead_1) * out_row_tiles + tile_row) * out_col_tiles + tile_col.
// Leading dims are padded on the left with 1 to two entries; padding is high-only, so the input
// tile at the same coordinates exists iff every coordinate is below the input's extent.
namespace g6_pad {

constexpr uint32_t kGeometryArgCount = 10;

template <uint32_t geometry_ct_base>
struct NewStorageWalk {
    static constexpr uint32_t out_lead_0 = get_compile_time_arg_val(geometry_ct_base + 0);
    static constexpr uint32_t out_lead_1 = get_compile_time_arg_val(geometry_ct_base + 1);
    static constexpr uint32_t in_lead_0 = get_compile_time_arg_val(geometry_ct_base + 2);
    static constexpr uint32_t in_lead_1 = get_compile_time_arg_val(geometry_ct_base + 3);
    static constexpr uint32_t out_row_tiles = get_compile_time_arg_val(geometry_ct_base + 4);
    static constexpr uint32_t out_col_tiles = get_compile_time_arg_val(geometry_ct_base + 5);
    static constexpr uint32_t in_row_tiles = get_compile_time_arg_val(geometry_ct_base + 6);
    static constexpr uint32_t in_col_tiles = get_compile_time_arg_val(geometry_ct_base + 7);
    // Logical rows / columns in the input's last tile row / column; 0 = no mask on that axis.
    static constexpr uint32_t row_mask_rem = get_compile_time_arg_val(geometry_ct_base + 8);
    static constexpr uint32_t col_mask_rem = get_compile_time_arg_val(geometry_ct_base + 9);
    static constexpr bool has_masks = row_mask_rem != 0 || col_mask_rem != 0;

    uint32_t lead_0 = 0;
    uint32_t lead_1 = 0;
    uint32_t tile_row = 0;
    uint32_t tile_col = 0;
    bool slice_exists = false;
    uint32_t in_slice_first_page = 0;

    void start(uint32_t w) {
        tile_col = w % out_col_tiles;
        const uint32_t rows = w / out_col_tiles;
        tile_row = rows % out_row_tiles;
        const uint32_t out_slice = rows / out_row_tiles;
        lead_1 = out_slice % out_lead_1;
        lead_0 = out_slice / out_lead_1;
        enter_slice();
    }

    void advance() {
        if (++tile_col < out_col_tiles) {
            return;
        }
        tile_col = 0;
        if (++tile_row < out_row_tiles) {
            return;
        }
        tile_row = 0;
        if (++lead_1 == out_lead_1) {
            lead_1 = 0;
            ++lead_0;
        }
        enter_slice();
    }

    bool has_input() const { return slice_exists && tile_row < in_row_tiles && tile_col < in_col_tiles; }

    uint32_t in_page() const { return in_slice_first_page + tile_row * in_col_tiles + tile_col; }

    // Valid only for an input-backed tile.
    bool row_mask() const { return row_mask_rem != 0 && tile_row == in_row_tiles - 1; }
    bool col_mask() const { return col_mask_rem != 0 && tile_col == in_col_tiles - 1; }

private:
    void enter_slice() {
        slice_exists = lead_0 < in_lead_0 && lead_1 < in_lead_1;
        in_slice_first_page = (lead_0 * in_lead_1 + lead_1) * in_row_tiles * in_col_tiles;
    }
};

}  // namespace g6_pad
