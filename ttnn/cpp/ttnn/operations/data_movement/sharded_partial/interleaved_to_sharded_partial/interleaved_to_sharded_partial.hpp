// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn {
namespace operations::data_movement {

struct InterleavedToShardedPartialOperation {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const std::variant<CoreCoord, CoreRangeSet>& grid,
        const std::array<uint32_t, 2>& shard_shape,
        int64_t& num_slices,
        int64_t& slice_index,
        tt::tt_metal::TensorMemoryLayout shard_scheme,
        tt::tt_metal::ShardOrientation shard_orientation,
        const std::optional<DataType>& data_type_arg);
};

}  // namespace operations::data_movement

constexpr auto interleaved_to_sharded_partial = ttnn::register_operation_with_auto_launch_op<
    "ttnn::interleaved_to_sharded_partial",
    ttnn::operations::data_movement::InterleavedToShardedPartialOperation>();
}  // namespace ttnn
