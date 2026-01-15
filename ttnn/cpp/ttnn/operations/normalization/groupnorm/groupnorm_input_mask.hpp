// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
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
ttnn::Tensor create_group_norm_input_mask(
    int64_t num_channel,
    int64_t num_groups,
    int64_t num_cores_across_channel,
    tt::tt_metal::DataType data_type = tt::tt_metal::DataType::BFLOAT16);

ttnn::Tensor create_group_norm_input_negative_mask(
    int64_t num_channel,
    int64_t num_groups,
    int64_t num_cores_across_channel,
    tt::tt_metal::DataType data_type = tt::tt_metal::DataType::BFLOAT16);
}  // namespace normalization
