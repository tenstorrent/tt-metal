// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn {
namespace operations::data_movement {

struct InterleavedToShardedOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const MemoryConfig& sharded_memory_config,
        const std::optional<DataType>& data_type_arg,
        const std::optional<bool>& keep_l1_aligned = std::nullopt);
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const std::variant<CoreCoord, CoreRangeSet>& grid,
        std::array<uint32_t, 2> shard_shape,
        TensorMemoryLayout shard_scheme,
        tt::tt_metal::ShardOrientation shard_orientation,
        const std::optional<DataType>& data_type_arg,
        const std::optional<bool>& keep_l1_aligned = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto interleaved_to_sharded = ttnn::register_operation<
    "ttnn::interleaved_to_sharded",
    ttnn::operations::data_movement::InterleavedToShardedOperation>();
}  // namespace ttnn
