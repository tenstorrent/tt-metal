// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_input_mask.hpp"
#include <algorithm>
#include "ttnn/types.hpp"

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;

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
        max_tile_span = std::max(current_tile_span, max_tile_span);
        current_position = group_end;
    }
    return max_tile_span;
}

ttnn::Tensor create_group_norm_input_mask_impl(
    int64_t num_channel,
    int64_t num_groups,
    int64_t num_cores_across_channel,
    DataType data_type,
    bool is_negative_mask,
    int64_t tile_height,
    int64_t tile_width,
    int64_t rows_in_last_tile) {
    TT_FATAL(num_cores_across_channel > 0, "create_group_norm_input_mask: num_cores_across_channel must be > 0.");
    TT_FATAL(
        num_groups % num_cores_across_channel == 0,
        "create_group_norm_input_mask: num_groups ({}) must be divisible by num_cores_across_channel ({}). "
        "The num_virtual_cols / num_cores_across_channel value must evenly divide both "
        "the channels into tiles and the number of groups.",
        num_groups,
        num_cores_across_channel);
    int64_t block_wt = find_max_tile_span(num_channel, num_channel / num_groups, tile_width);

    // rows_in_last_tile > 0 appends a second copy of every group with the padding rows zeroed.
    // Free here: the mask is assembled on host and uploaded once either way.
    TT_FATAL(
        rows_in_last_tile >= 0 && rows_in_last_tile < tile_height,
        "create_group_norm_input_mask: rows_in_last_tile ({}) must be in [0, tile_height={})",
        rows_in_last_tile,
        tile_height);
    const bool has_row_mask = rows_in_last_tile > 0;
    const int64_t mask_sets = has_row_mask ? 2 : 1;

    const int64_t out_num_groups = num_groups * mask_sets;
    const int64_t out_tile_height = tile_height;
    const int64_t out_mask_width = block_wt * tile_width;

    const int64_t num_groups_per_core = num_groups / num_cores_across_channel;
    const int64_t num_cols_per_group = num_channel / num_groups;

    std::vector<int64_t> start_strides;
    start_strides.reserve(num_cores_across_channel * num_groups_per_core);
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
    end_strides.reserve(start_strides.size());
    for (auto s : start_strides) {
        end_strides.push_back(s + num_cols_per_group);
    }

    const float mask_value = is_negative_mask ? 0.0f : 1.0f;
    std::vector<float> mask_vec(out_num_groups * out_tile_height * out_mask_width,
                                is_negative_mask ? 1.0f : 0.0f);

    for (int64_t group = 0; group < out_num_groups; ++group) {
        // Second set repeats the first but stops at rows_in_last_tile.
        const bool is_row_masked_set = group >= num_groups;
        const int64_t src_group = group % num_groups;
        const int64_t row_limit = is_row_masked_set ? rows_in_last_tile : out_tile_height;
        int64_t start_stride = start_strides[src_group];
        int64_t end_stride = std::min(end_strides[src_group], out_mask_width);
        const int64_t group_base = group * out_tile_height * out_mask_width;
        for (int64_t h = 0; h < row_limit; ++h) {
            const int64_t row_base = group_base + (h * out_mask_width);
            for (int64_t w = start_stride; w < end_stride; ++w) {
                mask_vec[row_base + w] = mask_value;
            }
        }
    }
    // create ttnn::Tensor from mask_vec
    const ttnn::Shape tensor_shape{1, out_num_groups, out_tile_height, out_mask_width};
    const tt::tt_metal::TensorLayout tensor_layout(data_type, Layout::TILE, ttnn::DRAM_MEMORY_CONFIG);
    const tt::tt_metal::TensorSpec tensor_spec(tensor_shape, tensor_layout);
    ttnn::Tensor mask = ttnn::Tensor::from_vector(
        mask_vec,
        tensor_spec,
        nullptr);

    return mask;
}

ttnn::Tensor create_group_norm_input_mask(
    int64_t num_channel,
    int64_t num_groups,
    int64_t num_cores_across_channel,
    DataType data_type,
    int64_t tile_height,
    int64_t tile_width,
    int64_t rows_in_last_tile) {
    return create_group_norm_input_mask_impl(
        num_channel,
        num_groups,
        num_cores_across_channel,
        data_type,
        false,
        tile_height,
        tile_width,
        rows_in_last_tile);
}

ttnn::Tensor create_group_norm_row_mask(
    int64_t rows_valid, int64_t num_groups, int64_t mask_width, DataType data_type, int64_t tile_height) {
    TT_FATAL(
        rows_valid > 0 && rows_valid < tile_height,
        "create_group_norm_row_mask: rows_valid ({}) must be in (0, tile_height={})",
        rows_valid,
        tile_height);
    TT_FATAL(num_groups > 0, "create_group_norm_row_mask: num_groups ({}) must be > 0", num_groups);
    std::vector<float> v(num_groups * tile_height * mask_width, 0.0f);
    for (int64_t g = 0; g < num_groups; ++g) {
        const int64_t group_base = g * tile_height * mask_width;
        for (int64_t h = 0; h < rows_valid; ++h) {
            const int64_t row_base = group_base + (h * mask_width);
            for (int64_t w = 0; w < mask_width; ++w) {
                v[row_base + w] = 1.0f;
            }
        }
    }
    const ttnn::Shape shape{
        1,
        static_cast<uint32_t>(num_groups),
        static_cast<uint32_t>(tile_height),
        static_cast<uint32_t>(mask_width)};
    const tt::tt_metal::TensorLayout layout(data_type, Layout::TILE, ttnn::DRAM_MEMORY_CONFIG);
    return ttnn::Tensor::from_vector(v, tt::tt_metal::TensorSpec(shape, layout), nullptr);
}

ttnn::Tensor create_group_norm_input_negative_mask(
    int64_t num_channel,
    int64_t num_groups,
    int64_t num_cores_across_channel,
    DataType data_type,
    int64_t tile_height,
    int64_t tile_width) {
    return create_group_norm_input_mask_impl(
        num_channel, num_groups, num_cores_across_channel, data_type, true, tile_height, tile_width, 0);
}
}  // namespace normalization
