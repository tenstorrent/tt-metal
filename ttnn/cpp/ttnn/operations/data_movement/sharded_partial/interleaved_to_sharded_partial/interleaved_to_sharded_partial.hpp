// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn {

ttnn::Tensor interleaved_to_sharded_partial(
    const ttnn::Tensor& input_tensor,
    const std::variant<CoreCoord, CoreRangeSet>& grid,
    const std::array<uint32_t, 2>& shard_shape,
    int64_t& num_slices,
    int64_t& slice_index,
    tt::tt_metal::TensorMemoryLayout shard_scheme,
    tt::tt_metal::ShardOrientation shard_orientation,
    const std::optional<DataType>& data_type_arg = std::nullopt);

}  // namespace ttnn
