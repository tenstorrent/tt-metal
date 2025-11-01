// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_input_mask.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::constants;

namespace ttnn::operations::normalization {

// Finds the maximum (worst case) number of tiles a group of size group_size can span across.
// This helps in setting the mask width conservatively.
static int64_t find_max_tile_span(int64_t W, int64_t group_size, int64_t tile_width) {
    TT_FATAL(W > 0, "W needs to be greater than 0 and is {}", W);
    TT_FATAL(group_size > 0, "group_size needs to be greater than 0 and is {}", group_size);

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

ttnn::Tensor create_group_norm_input_mask_impl(int64_t num_channel, int64_t num_groups,
      int64_t num_cores_across_channel, DataType data_type, bool is_negative_mask) {
    int64_t block_wt = find_max_tile_span(num_channel, num_channel / num_groups, TILE_WIDTH);

    const int64_t out_num_groups = num_groups;
    const int64_t out_tile_height = TILE_HEIGHT;
    const int64_t out_mask_width = block_wt * TILE_WIDTH;

    const int64_t num_groups_per_core = num_groups / num_cores_across_channel;
    const int64_t num_cols_per_group = num_channel / num_groups;

    std::vector<int64_t> start_strides;
    for (int64_t core = 0; core < num_cores_across_channel; ++core) {
        int64_t row_offset = 0;
        start_strides.push_back(0);
        for (int64_t group = 0; group < num_groups_per_core - 1; ++group) {
            if (row_offset + (num_cols_per_group % TILE_WIDTH) == TILE_WIDTH) {
                row_offset = 0;
            } else if (row_offset + (num_cols_per_group % TILE_WIDTH) > TILE_WIDTH) {
                row_offset = (num_cols_per_group % TILE_WIDTH) + row_offset - TILE_WIDTH;
            } else {
                row_offset += num_cols_per_group % TILE_WIDTH;
            }
            start_strides.push_back(row_offset);
        }
    }
    std::vector<int64_t> end_strides;
    end_strides.reserve(start_strides.size());
    for (auto s : start_strides) {
        end_strides.push_back(s + num_cols_per_group);
    }

    const float mask_value = is_negative_mask ? 0.0f : 1.0f;
    std::vector<float> mask_vec(out_num_groups * out_tile_height * out_mask_width,
                                is_negative_mask ? 1.0f : 0.0f);

    for (int64_t group = 0; group < out_num_groups; ++group) {
        int64_t start_stride = start_strides[group];
        int64_t end_stride = std::min(end_strides[group], out_mask_width);
        for (int64_t h = 0; h < out_tile_height; ++h) {
            for (int64_t w = start_stride; w < end_stride; ++w) {
                int64_t idx = (group * out_tile_height * out_mask_width) + (h * out_mask_width) + w;
                mask_vec[idx] = mask_value;
            }
        }
    }
    // create ttnn::Tensor from mask_vec
    const ttnn::Shape tensor_shape{1, out_num_groups, out_tile_height, out_mask_width};
    const tt::tt_metal::TensorLayout tensor_layout(data_type, Layout::TILE, ttnn::DRAM_MEMORY_CONFIG);
    const ttnn::TensorSpec tensor_spec(tensor_shape, tensor_layout);
    ttnn::Tensor mask = ttnn::Tensor::from_vector(
        mask_vec,
        tensor_spec,
        nullptr);

    return mask;
}

ttnn::Tensor create_group_norm_input_mask(int64_t num_channel, int64_t num_groups,
      int64_t num_cores_across_channel, DataType data_type) {
    return create_group_norm_input_mask_impl(num_channel, num_groups, num_cores_across_channel, data_type, false);
}

ttnn::Tensor create_group_norm_input_negative_mask(
    int64_t num_channel, int64_t num_groups, int64_t num_cores_across_channel, DataType data_type) {
    return create_group_norm_input_mask_impl(num_channel, num_groups, num_cores_across_channel, data_type, true);
}
}  // namespace normalization
