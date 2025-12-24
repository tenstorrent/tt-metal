// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
ttnn::operations::data_movement::ShardedToInterleavedDeviceOperation::tensor_return_value_t sharded_to_interleaved(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    const std::optional<Tensor>& preallocated_output) {
    using OperationType = ttnn::operations::data_movement::ShardedToInterleavedDeviceOperation;
    return ttnn::device_operation::detail::launch_on_device<OperationType>(
        OperationType::operation_attributes_t{
            .output_mem_config = output_mem_config,
            .output_dtype = output_dtype,
            .num_slices = 1,
            .slice_index = 0},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
            .preallocated_output = preallocated_output});
}
}  // namespace ttnn::prim

namespace ttnn::operations::data_movement {

}  // namespace ttnn::operations::data_movement
