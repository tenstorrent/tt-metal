// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "device/interleaved_to_sharded_op.hpp"
#include "interleaved_to_sharded.hpp"
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor InterleavedToShardedOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& sharded_memory_config,
    const std::optional<DataType>& data_type_arg,
    const std::optional<bool>& keep_l1_aligned) {
    return operation::run(
               InterleavedToShardedDeviceOperation{
                   .output_mem_config = sharded_memory_config,
                   .output_dtype = data_type_arg.value_or(input_tensor.get_dtype()),
                   .keep_l1_aligned = keep_l1_aligned.value_or(false)},
               {input_tensor})
        .at(0);
}

ttnn::Tensor InterleavedToShardedOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const std::variant<CoreCoord, CoreRangeSet>& grid,
    const std::array<uint32_t, 2> shard_shape,
    const TensorMemoryLayout shard_scheme,
    const ShardOrientation shard_orientation,
    const std::optional<DataType>& data_type_arg,
    const std::optional<bool>& keep_l1_aligned) {
    bool row_wise = shard_orientation == ShardOrientation::ROW_MAJOR;
    CoreCoord grid_size;
    CoreRangeSet grid_set;
    std::visit(
        [&](const auto& grid) {
            using GridType = std::decay_t<decltype(grid)>;
            if constexpr (std::is_same_v<GridType, CoreCoord>) {
                grid_size = grid;
                uint32_t num_cores = 0;
                uint32_t total_height = input_tensor.volume() / input_tensor.get_padded_shape()[-1];
                uint32_t total_width = input_tensor.get_padded_shape()[-1];
                switch (shard_scheme) {
                    case TensorMemoryLayout::HEIGHT_SHARDED:
                        num_cores = tt::div_up(total_height, shard_shape[0]);
                        break;
                    case TensorMemoryLayout::WIDTH_SHARDED: num_cores = tt::div_up(total_width, shard_shape[1]); break;
                    case TensorMemoryLayout::BLOCK_SHARDED:
                        num_cores = tt::div_up(total_height, shard_shape[0]) * tt::div_up(total_width, shard_shape[1]);
                        break;
                    default: TT_ASSERT(false, "Unsupported sharding scheme");
                }
                grid_set = tt::tt_metal::num_cores_to_corerangeset(num_cores, grid_size, row_wise);
            } else if constexpr (std::is_same_v<GridType, CoreRangeSet>) {
                auto bbox = grid.bounding_box();
                grid_size = CoreCoord{bbox.end_coord.x + 1, bbox.end_coord.y + 1};
                grid_set = grid;
            }
        },
        grid);
    ShardSpec shard_spec(grid_set, shard_shape, shard_orientation);
    MemoryConfig sharded_mem_config = MemoryConfig{shard_scheme, BufferType::L1, shard_spec};

    return operation::run(
               InterleavedToShardedDeviceOperation{
                   .output_mem_config = sharded_mem_config,
                   .output_dtype = data_type_arg.value_or(input_tensor.get_dtype()),
                   .keep_l1_aligned = keep_l1_aligned.value_or(false)},
               {input_tensor})
        .at(0);
}

}  // namespace ttnn::operations::data_movement
