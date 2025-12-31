// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard.hpp"
#include "device/reshard_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {

Tensor reshard(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    using OperationType = operations::data_movement::reshard::ReshardDeviceOperation;
    return device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_mem_config = memory_config,
        },
        OperationType::tensor_args_t{.input = input_tensor, .preallocated_output = optional_output_tensor});
}

}  // namespace ttnn
