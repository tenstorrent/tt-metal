// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "intimg_work_split.hpp"

#include <iostream>

namespace ttnn::operations::experimental::reduction::intimg::common {

CoreRangeSet generate_max_core_range_set(
    const Shape& input_shape, const CoreCoord& core_grid_size, uint32_t tile_height, uint32_t tile_width) {
    std::cout << __FUNCTION__ << std::endl;
    const uint32_t num_channels = input_shape[3];
    const uint32_t input_height = input_shape[2];
    const std::size_t num_slices_along_channels = (num_channels + tile_width - 1) / tile_width;
    const std::size_t num_blocks_in_column = (input_height + tile_height - 1) / tile_height;

    const uint32_t engaged_cores_along_channels = std::min(num_slices_along_channels, core_grid_size.x);
    const uint32_t engaged_cores_along_rows = std::min(num_blocks_in_column, core_grid_size.y);
    const uint32_t END_X = engaged_cores_along_channels - 1;
    const uint32_t END_Y = engaged_cores_along_rows - 1;

    std::cout << __FUNCTION__ << std::endl
              << num_channels << std::endl
              << input_height << std::endl
              << num_slices_along_channels << std::endl
              << num_blocks_in_column << std::endl
              << engaged_cores_along_channels << std::endl
              << engaged_cores_along_rows << std::endl
              << END_X << std::endl
              << END_Y << std::endl
              << std::endl;

    return CoreRangeSet{CoreRange{CoreCoord{START_X, START_Y}, CoreCoord{END_X, END_Y}}};
}

IntImgPerCoreSetWorkSplit split_intimg_work_to_cores(
    const Shape& input_shape, const CoreCoord& core_grid_size, uint32_t tile_height, uint32_t tile_width) {
    std::cout << __FUNCTION__ << std::endl;
    const uint32_t row_chunks_along_channels = (input_shape[3] + tile_width - 1) / tile_width;
    const uint32_t row_chunks_along_height = (input_shape[2] + tile_height - 1) / tile_height;
    const uint32_t cores_along_channels = core_grid_size.x;
    const uint32_t cores_along_height = core_grid_size.y;
    uint32_t left_cores_channel_chunks;
    uint32_t right_cores_channel_chunks;
    uint32_t top_cores_height_chunks;
    uint32_t bottom_cores_height_chunks;
    if (row_chunks_along_channels % cores_along_channels == 0) {
        left_cores_channel_chunks = row_chunks_along_channels / cores_along_channels;
        right_cores_channel_chunks = 0;
    } else {
        left_cores_channel_chunks = (row_chunks_along_channels / cores_along_channels) + 1;
        right_cores_channel_chunks = row_chunks_along_channels / cores_along_channels;
    }
    if (row_chunks_along_height % cores_along_height == 0) {
        top_cores_height_chunks = row_chunks_along_height / cores_along_height;
        bottom_cores_height_chunks = 0;
    } else {
        top_cores_height_chunks = (row_chunks_along_height / cores_along_height) + 1;
        bottom_cores_height_chunks = row_chunks_along_height / cores_along_height;
    }
    const uint32_t left_cores_end_x = row_chunks_along_channels % cores_along_channels;
    const uint32_t right_cores_begin_x = left_cores_end_x + 1;
    const uint32_t right_cores_end_x = cores_along_channels - 1;
    const uint32_t top_cores_end_y = row_chunks_along_height % cores_along_height;
    const uint32_t bottom_cores_begin_y = top_cores_end_y + 1;
    const uint32_t bottom_cores_end_y = cores_along_height - 1;
    const uint32_t row_chunks_along_channels_per_left_cores = (row_chunks_along_channels / cores_along_channels) + 1;
    const uint32_t row_chunks_along_channels_per_right_cores = (row_chunks_along_channels / cores_along_channels);
    const uint32_t row_chunks_along_height_per_top_cores = (row_chunks_along_height / cores_along_height) + 1;
    const uint32_t row_chunks_along_height_per_bottom_cores = (row_chunks_along_height / cores_along_height);
    const uint32_t starting_row_chunk_along_channels_right =
        row_chunks_along_channels_per_left_cores * left_cores_end_x;
    const uint32_t starting_row_chunk_along_height_bottom = row_chunks_along_height_per_top_cores * top_cores_end_y;
    const bool top_left_core_set_engaged =
        (row_chunks_along_channels > 0) && (row_chunks_along_height > 0);  // should be always true
    const bool top_right_core_set_engaged =
        (row_chunks_along_channels > cores_along_channels) && (left_cores_end_x < (cores_along_channels - 1));
    const bool bottom_left_core_set_engaged =
        (row_chunks_along_height > cores_along_height) && (top_cores_end_y < (cores_along_height - 1));
    const bool bottom_right_core_set_engaged = top_right_core_set_engaged && bottom_left_core_set_engaged;
    std::cout << __FUNCTION__ << std::endl
              << row_chunks_along_channels << std::endl
              << row_chunks_along_height << std::endl
              << cores_along_channels << std::endl
              << cores_along_height << std::endl
              << left_cores_channel_chunks << std::endl
              << right_cores_channel_chunks << std::endl
              << top_cores_height_chunks << std::endl
              << bottom_cores_height_chunks << std::endl
              << left_cores_end_x << std::endl
              << right_cores_begin_x << std::endl
              << right_cores_end_x << std::endl
              << top_cores_end_y << std::endl
              << bottom_cores_begin_y << std::endl
              << bottom_cores_end_y << std::endl
              << row_chunks_along_channels_per_left_cores << std::endl
              << row_chunks_along_channels_per_right_cores << std::endl
              << row_chunks_along_height_per_top_cores << std::endl
              << row_chunks_along_height_per_bottom_cores << std::endl
              << starting_row_chunk_along_channels_right << std::endl
              << starting_row_chunk_along_height_bottom << std::endl
              << top_left_core_set_engaged << std::endl
              << top_right_core_set_engaged << std::endl
              << bottom_left_core_set_engaged << std::endl
              << bottom_right_core_set_engaged << std::endl
              << std::endl;
    return {
        .top_left =
            {top_left_core_set_engaged,
             {CoreRange{{START_X, START_Y}, {left_cores_end_x, top_cores_end_y}},
              {0, row_chunks_along_channels_per_left_cores, 0, row_chunks_along_height_per_top_cores}}},
        .bottom_left =
            {bottom_left_core_set_engaged,
             {CoreRange{{START_X, bottom_cores_begin_y}, {left_cores_end_x, bottom_cores_end_y}},
              {0,
               row_chunks_along_channels_per_left_cores,
               starting_row_chunk_along_height_bottom,
               row_chunks_along_height_per_bottom_cores}}},
        .top_right =
            {top_right_core_set_engaged,
             {CoreRange{{right_cores_begin_x, START_Y}, {right_cores_end_x, top_cores_end_y}},
              {starting_row_chunk_along_channels_right,
               row_chunks_along_channels_per_right_cores,
               0,
               row_chunks_along_height_per_top_cores}}},
        .bottom_right =
            {bottom_right_core_set_engaged,
             {CoreRange{{right_cores_begin_x, bottom_cores_begin_y}, {right_cores_end_x, bottom_cores_end_y}},
              {starting_row_chunk_along_channels_right,
               row_chunks_along_channels_per_right_cores,
               starting_row_chunk_along_height_bottom,
               row_chunks_along_height_per_bottom_cores}}},
    };
}

IntImgPerCoreSetWorkSplitMap make_intimg_work_map(const IntImgPerCoreSetWorkSplit& work_split) {
    return {
        {"top_left", work_split.top_left},
        {"bottom_left", work_split.bottom_left},
        {"top_right", work_split.top_right},
        {"bottom_right", work_split.bottom_right},
    };
}

}  // namespace ttnn::operations::experimental::reduction::intimg::common
