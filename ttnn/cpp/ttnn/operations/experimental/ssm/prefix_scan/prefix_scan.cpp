// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prefix_scan.hpp"
#include "device/prefix_scan_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental {

Tensor prefix_scan(
    const Tensor& a,
    const Tensor& bx,
    const Tensor& h_prev,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> dtype,
    std::optional<MathFidelity> math_fidelity) {
    using OperationType = operations::experimental::ssm::prefix_scan::PrefixScanDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .memory_config = memory_config.value_or(a.memory_config()),
        .dtype = dtype.value_or(a.dtype()),
        .math_fidelity = math_fidelity.value_or(MathFidelity::HiFi4),
    };
    auto tensor_args = OperationType::tensor_args_t{.a = a, .bx = bx, .h_prev = h_prev};

    return device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
