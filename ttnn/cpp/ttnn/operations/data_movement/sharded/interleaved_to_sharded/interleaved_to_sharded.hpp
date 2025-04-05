// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/core_coord.hpp>
#include <array>
#include <optional>
#include <variant>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
enum class DataType;
enum class ShardOrientation;
enum class TensorMemoryLayout;
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

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
        const std::array<uint32_t, 2> shard_shape,
        const TensorMemoryLayout shard_scheme,
        const tt::tt_metal::ShardOrientation shard_orientation,
        const std::optional<DataType>& data_type_arg,
        const std::optional<bool>& keep_l1_aligned = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto interleaved_to_sharded = ttnn::register_operation_with_auto_launch_op<
    "ttnn::interleaved_to_sharded",
    ttnn::operations::data_movement::InterleavedToShardedOperation>();
}  // namespace ttnn
