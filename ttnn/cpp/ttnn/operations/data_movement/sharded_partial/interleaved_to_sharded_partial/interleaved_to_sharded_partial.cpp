// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"
#include "ttnn/run_operation.hpp"
#include "device/interleaved_to_sharded_partial_op.hpp"
#include "interleaved_to_sharded_partial.hpp"
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor InterleavedToShardedPartialOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const std::variant<CoreCoord, CoreRangeSet>& grid,
    const std::array<uint32_t, 2>& shard_shape,
    int64_t& num_slices,
    int64_t& slice_index,
    tt::tt_metal::TensorMemoryLayout shard_scheme,
    tt::tt_metal::ShardOrientation shard_orientation,
    const std::optional<DataType>& data_type_arg) {
    bool row_wise = shard_orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR;
    CoreCoord grid_size;
    CoreRangeSet grid_set;
    std::visit(
        [&](const auto& grid) {
            using GridType = std::decay_t<decltype(grid)>;
            if constexpr (std::is_same_v<GridType, CoreCoord>) {
                grid_size = grid;
                uint32_t num_cores = 0;
                uint32_t total_height = input_tensor.volume() / input_tensor.get_padded_shape()[-1];
                total_height /= num_slices;

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
                TT_THROW("Unsupported type for grid. CoreRangeSet not supported. Switch to a different type.");
            }
        },
        grid);

    tt::tt_metal::ShardSpec shard_spec(grid_set, shard_shape, shard_orientation);
    tt::tt_metal::MemoryConfig sharded_mem_config = tt::tt_metal::MemoryConfig{shard_scheme, BufferType::L1};
    return operation::run(
               InterleavedToShardedPartialDeviceOperation{
                   .grid_size = grid_size,
                   .shard_spec = shard_spec,
                   .num_slices = num_slices,
                   .slice_index = slice_index,
                   .output_mem_config = sharded_mem_config,
                   .output_dtype = data_type_arg.value_or(input_tensor.get_dtype())},
               {input_tensor})
        .at(0);
}

}  // namespace ttnn::operations::data_movement
