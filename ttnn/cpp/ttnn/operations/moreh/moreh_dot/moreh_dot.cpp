// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot.hpp"

#include "ttnn/operations/moreh/moreh_dot/device/moreh_dot_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::moreh::moreh_dot {

Tensor moreh_dot(
    const Tensor& input_a,
    const Tensor& input_b,
    const std::optional<Tensor>& output,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = MorehDotOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        dtype.value_or(input_a.dtype()),
        memory_config.value_or(input_a.memory_config()),
        init_device_compute_kernel_config(input_a.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{input_a, input_b, output};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::moreh::moreh_dot
