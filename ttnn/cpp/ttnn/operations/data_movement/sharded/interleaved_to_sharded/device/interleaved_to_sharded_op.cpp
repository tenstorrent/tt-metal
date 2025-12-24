// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_op.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include <ttnn/operation.hpp>

namespace ttnn::prim {
ttnn::operations::data_movement::InterleavedToShardedDeviceOperation::tensor_return_value_t interleaved_to_sharded(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    bool keep_l1_aligned,
    const std::optional<Tensor>& preallocated_output) {
    using OperationType = ttnn::operations::data_movement::InterleavedToShardedDeviceOperation;
    return ttnn::device_operation::detail::launch_on_device<OperationType>(
        OperationType::operation_attributes_t{output_mem_config, output_dtype, keep_l1_aligned},
        OperationType::tensor_args_t{input_tensor, preallocated_output});
}
}  // namespace ttnn::prim

namespace ttnn::operations::data_movement {

}  // namespace ttnn::operations::data_movement
