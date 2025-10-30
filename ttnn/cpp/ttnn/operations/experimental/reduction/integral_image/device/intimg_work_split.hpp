// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/shape.hpp"

#include <iostream>
#include <map>
#include <string>

namespace ttnn::operations::experimental::reduction::intimg::common {

using namespace tt::tt_metal;

constexpr uint32_t START_X = 0;
constexpr uint32_t START_Y = 0;

// struct IntImgPerCoreWorkDef {
//     const uint32_t starting_row_chunk_along_channels;
//     const uint32_t row_chunks_along_channels;
//     const uint32_t starting_row_chunk_along_height;
//     const uint32_t row_chunks_along_height;
// };

struct IntImgPerCoreSetWorkDef {
    const uint32_t starting_row_chunk_per_core_set_along_channels;
    const uint32_t row_chunks_per_core_along_channels;
    const uint32_t starting_row_chunk_per_core_set_along_height;
    const uint32_t row_chunks_per_core_along_height;
};

using CoreRangeSetEngaged = bool;
using IntImgWorkDefPerCoreSet = std::pair<CoreRangeSetEngaged, std::pair<CoreRangeSet, IntImgPerCoreSetWorkDef>>;

struct IntImgPerCoreSetWorkSplit {
    const IntImgWorkDefPerCoreSet top_left;
    const IntImgWorkDefPerCoreSet bottom_left;
    const IntImgWorkDefPerCoreSet top_right;
    const IntImgWorkDefPerCoreSet bottom_right;
};

using IntImgPerCoreSetWorkSplitMap = std::map<std::string, IntImgWorkDefPerCoreSet>;

constexpr uint32_t EXPECTED_AVAILABLE_CORES_IN_ROW = 5;
constexpr uint32_t EXPECTED_AVAILABLE_CORES_IN_COLUMN = 4;

CoreRangeSet generate_max_core_range_set(
    const Shape& input_shape,
    const CoreCoord& core_grid_size = {EXPECTED_AVAILABLE_CORES_IN_ROW, EXPECTED_AVAILABLE_CORES_IN_COLUMN},
    uint32_t tile_height = 32,
    uint32_t tile_width = 32);

IntImgPerCoreSetWorkSplit split_intimg_work_to_cores(
    const Shape& input_shape,
    const CoreCoord& core_grid_size = {EXPECTED_AVAILABLE_CORES_IN_ROW, EXPECTED_AVAILABLE_CORES_IN_COLUMN},
    uint32_t tile_height = 32,
    uint32_t tile_width = 32);

IntImgPerCoreSetWorkSplitMap make_intimg_work_map(const IntImgPerCoreSetWorkSplit& work_split);

}  // namespace ttnn::operations::experimental::reduction::intimg::common
