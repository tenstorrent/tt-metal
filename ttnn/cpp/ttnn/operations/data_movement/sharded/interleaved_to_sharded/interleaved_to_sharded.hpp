// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "tt_metal/common/core_coord.h"


namespace ttnn {
namespace operations::data_movement {

struct InterleavedToShardedOperation {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const MemoryConfig& sharded_memory_config,
        const std::optional<DataType> & data_type_arg
        );
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const std::variant<CoreCoord, CoreRangeSet> grid,
        const std::array<uint32_t, 2> shard_shape,
        const TensorMemoryLayout shard_scheme,
        const ShardOrientation shard_orientation,
        const std::optional<DataType> & data_type_arg
        );

};


}  // namespace operations::data_movement

constexpr auto interleaved_to_sharded = ttnn::register_operation_with_auto_launch_op<"ttnn::interleaved_to_sharded", ttnn::operations::data_movement::InterleavedToShardedOperation>();
}  // namespace ttnn
