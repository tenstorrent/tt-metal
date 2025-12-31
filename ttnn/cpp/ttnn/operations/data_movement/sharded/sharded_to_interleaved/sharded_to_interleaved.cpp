// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_to_interleaved.hpp"
#include "device/sharded_to_interleaved_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {

Tensor sharded_to_interleaved(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    const std::optional<Tensor>& preallocated_output) {
    // Early return if input is not sharded
    if (!input_tensor.shard_spec().has_value()) {
        return input_tensor;
    }

    const auto resolved_dtype = output_dtype.value_or(input_tensor.dtype());
    using OperationType = operations::data_movement::ShardedToInterleavedDeviceOperation;
    return device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_mem_config = output_mem_config, .output_dtype = resolved_dtype, .num_slices = 1, .slice_index = 0},
        OperationType::tensor_args_t{.input_tensor = input_tensor, .preallocated_output = preallocated_output});
}

}  // namespace ttnn
