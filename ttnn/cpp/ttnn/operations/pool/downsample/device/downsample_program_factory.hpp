// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/operation.hpp"

// TODO: DELETE Do not use using namespace in header file
using namespace tt::constants;

namespace ttnn::operations::downsample::detail {

std::pair<uint32_t, uint32_t> get_num_cores_height_width_sliced(
    const CoreRangeSet& all_cores,
    tt::tt_metal::TensorMemoryLayout memory_layout,
    tt::tt_metal::ShardOrientation shard_orientation);
tt::tt_metal::operation::ProgramWithCallbacks downsample_single_core(
    const Tensor& a, std::array<uint32_t, 5> downsample_params, Tensor& output);

}  // namespace ttnn::operations::downsample::detail
