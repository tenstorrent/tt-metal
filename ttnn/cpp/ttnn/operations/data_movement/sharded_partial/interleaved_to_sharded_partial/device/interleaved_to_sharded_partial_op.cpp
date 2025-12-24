// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_partial_op.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <ttnn/operation.hpp>

namespace ttnn::prim {
ttnn::operations::data_movement::InterleavedToShardedPartialDeviceOperation::tensor_return_value_t
interleaved_to_sharded_partial(
    const Tensor& input_tensor,
    const CoreCoord& grid_size,
    const tt::tt_metal::ShardSpec& shard_spec,
    uint32_t num_slices,
    uint32_t slice_index,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype) {
    using OperationType = ttnn::operations::data_movement::InterleavedToShardedPartialDeviceOperation;
    return ttnn::device_operation::detail::launch_on_device<OperationType>(
        OperationType::operation_attributes_t{
            .grid_size = grid_size,
            .shard_spec = shard_spec,
            .num_slices = num_slices,
            .slice_index = slice_index,
            .output_mem_config = output_mem_config,
            .output_dtype = output_dtype},
        OperationType::tensor_args_t{.input_tensor = input_tensor});
}
}  // namespace ttnn::prim

namespace ttnn::operations::data_movement {

}  // namespace ttnn::operations::data_movement
