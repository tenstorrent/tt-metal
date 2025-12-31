// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dropout.hpp"
#include "device/dropout_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental {

Tensor dropout(const Tensor& input_tensor, float prob, float scale, uint32_t seed, bool use_per_device_seed) {
    using OperationType = operations::experimental::dropout::DropoutDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .output_dtype = DataType::BFLOAT16,
        .output_memory_config = MemoryConfig(),
        .seed = seed,
        .use_per_device_seed = use_per_device_seed,
        .prob = prob,
        .scale = scale,
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input_tensor, .preallocated_output = std::nullopt};

    return device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
