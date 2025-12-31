// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hc_sum_reduce.hpp"
#include "device/hc_sum_reduce_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental {

Tensor hc_sum_reduce(
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> dtype,
    std::optional<MathFidelity> math_fidelity) {
    using OperationType = operations::experimental::ssm::hc_sum_reduce::HCSumReduceDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .memory_config = memory_config.value_or(input.memory_config()),
        .dtype = dtype.value_or(input.dtype()),
        .math_fidelity = math_fidelity.value_or(MathFidelity::HiFi4),
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input};

    return device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
