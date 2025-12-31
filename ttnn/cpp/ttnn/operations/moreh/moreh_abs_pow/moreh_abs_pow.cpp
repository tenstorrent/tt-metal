// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_abs_pow.hpp"

#include "ttnn/operations/moreh/moreh_abs_pow/device/moreh_abs_pow_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::moreh::moreh_abs_pow {

Tensor moreh_abs_pow(
    const Tensor& input,
    const float p,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = MorehAbsPowOperation;
    const OperationType::operation_attributes_t operation_attributes{
        p,
        memory_config.value_or(input.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    const OperationType::tensor_args_t tensor_args{input, output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::moreh::moreh_abs_pow
