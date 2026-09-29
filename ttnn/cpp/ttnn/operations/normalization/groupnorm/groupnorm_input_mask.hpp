// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::normalization {

// C++ implementation of create_group_norm_input_mask.
// Create 4D mask [1, num_groups, 32, 32*block_wt] used by group norm.
// block_wt is computed from worst-case tile span across groups.
// num_cores_across_channel splits groups evenly across cores (must divide num_groups).
//
// rows_in_last_tile (= logical_hw % 32) is for non-tile-aligned H*W. It appends a second set of
// groups -- shape becomes [1, 2*num_groups, 32, 32*block_wt] -- identical to the first but with
// rows >= rows_in_last_tile zeroed; group_norm selects it on each batch's final row-tile. Leave at
// 0 for tile-aligned H*W. Callers supplying their own mask to a non-tile-aligned group_norm should
// set it, else group_norm derives the second set with a device-side multiply+concat per call.
ttnn::Tensor create_group_norm_input_mask(
    int64_t num_channel,
    int64_t num_groups,
    int64_t num_cores_across_channel,
    tt::tt_metal::DataType data_type = tt::tt_metal::DataType::BFLOAT16,
    int64_t tile_height = 32,
    int64_t tile_width = 32,
    int64_t rows_in_last_tile = 0);

// [1, num_groups, tile_height, mask_width], 1.0 on rows [0, rows_valid) and 0.0 above, identical
// per group. Multiplying the input mask by this gives the variant used on the final row-tile.
// Replicated across dim 1 so the multiply is plain elementwise and stays in the mask's dtype.
// Only the fallback for a mask not built with create_group_norm_input_mask's rows_in_last_tile.
ttnn::Tensor create_group_norm_row_mask(
    int64_t rows_valid,
    int64_t num_groups,
    int64_t mask_width,
    tt::tt_metal::DataType data_type = tt::tt_metal::DataType::BFLOAT16,
    int64_t tile_height = 32);

ttnn::Tensor create_group_norm_input_negative_mask(
    int64_t num_channel,
    int64_t num_groups,
    int64_t num_cores_across_channel,
    tt::tt_metal::DataType data_type = tt::tt_metal::DataType::BFLOAT16,
    int64_t tile_height = 32,
    int64_t tile_width = 32);
}  // namespace normalization
