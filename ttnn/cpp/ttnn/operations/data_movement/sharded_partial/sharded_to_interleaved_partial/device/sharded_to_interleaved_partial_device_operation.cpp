// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
ttnn::operations::data_movement::ShardedToInterleavedPartialDeviceOperation::tensor_return_value_t
sharded_to_interleaved_partial(
    const Tensor& input_tensor,
    const Tensor& cache_tensor,
    uint32_t num_slices,
    uint32_t slice_index,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype) {
    using OperationType = ttnn::operations::data_movement::ShardedToInterleavedPartialDeviceOperation;
    return ttnn::device_operation::detail::launch_on_device<OperationType>(
        OperationType::operation_attributes_t{
            .num_slices = num_slices,
            .slice_index = slice_index,
            .output_mem_config = output_mem_config,
            .output_dtype = output_dtype},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
            .cache_tensor = cache_tensor});
}
}  // namespace ttnn::prim

namespace ttnn::operations::data_movement {

}  // namespace ttnn::operations::data_movement
