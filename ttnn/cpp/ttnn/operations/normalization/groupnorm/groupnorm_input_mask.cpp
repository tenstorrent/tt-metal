// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_input_mask.hpp"

namespace ttnn::operations::normalization {

// Finds the maximum (worst case) number of tiles a group of size group_size can span across.
// This helps in setting the mask width conservatively.
static int64_t find_max_tile_span(int64_t W, int64_t group_size, int64_t tile_width) {
    int64_t current_position = 0;
    int64_t max_tile_span = 0;
    while (current_position < W) {
        int64_t group_end = current_position + group_size;
        int64_t start_tile = current_position / tile_width;
        int64_t end_tile = (group_end - 1) / tile_width;
        int64_t current_tile_span = end_tile - start_tile + 1;
        if (current_tile_span > max_tile_span) {
            max_tile_span = current_tile_span;
        }
        current_position = group_end;
    }
    return max_tile_span;
}

std::vector<float> create_group_norm_input_mask_impl(int64_t num_channel, int64_t num_groups,
      int64_t num_cores_across_channel, int64_t& out_num_groups, int64_t& out_tile_height,
      int64_t& out_mask_width) {
    constexpr int64_t tile_width = 32;
    constexpr int64_t tile_height = 32;
    int64_t block_wt = find_max_tile_span(num_channel, num_channel / num_groups, tile_width);

    out_num_groups = num_groups;
    out_tile_height = tile_height;
    out_mask_width = block_wt * tile_width;

    int64_t num_groups_per_core = num_groups / num_cores_across_channel;
    int64_t num_cols_per_group = num_channel / num_groups;

    std::vector<int64_t> start_strides;
    for (int64_t core = 0; core < num_cores_across_channel; ++core) {
        int64_t row_offset = 0;
        start_strides.push_back(0);
        for (int64_t group = 0; group < num_groups_per_core - 1; ++group) {
            if (row_offset + (num_cols_per_group % tile_width) == tile_width) {
                row_offset = 0;
            } else if (row_offset + (num_cols_per_group % tile_width) > tile_width) {
                row_offset = (num_cols_per_group % tile_width) + row_offset - tile_width;
            } else {
                row_offset += num_cols_per_group % tile_width;
            }
            start_strides.push_back(row_offset);
        }
    }
    std::vector<int64_t> end_strides;
    for (auto s : start_strides) end_strides.push_back(s + num_cols_per_group);

    std::vector<float> mask_vec(out_num_groups * out_tile_height * out_mask_width, 0.0f);
    for (int64_t group = 0; group < out_num_groups; ++group) {
        int64_t start_stride = start_strides[group];
        int64_t end_stride = std::min(end_strides[group], out_mask_width);
        for (int64_t h = 0; h < out_tile_height; ++h) {
            for (int64_t w = start_stride; w < end_stride; ++w) {
                int64_t idx = group * out_tile_height * out_mask_width + h * out_mask_width + w;
                mask_vec[idx] = 1.0f;
            }
        }
    }
    return mask_vec;
}
}  // namespace normalization
