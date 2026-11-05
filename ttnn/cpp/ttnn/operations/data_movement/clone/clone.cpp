// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "clone.hpp"

#include "device/clone_device_operation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttnn {

namespace {
bool clone_cannot_convert_dtype(const DataType& input_dtype, const DataType& output_dtype) {
    // The clone compute kernel copies tile data between differently formatted DFBs without
    // converting it, so widening a uint16 source (or converting it to float) returns shifted or
    // garbage values. Route those through ttnn::typecast, which converts correctly.
    return input_dtype == DataType::UINT16 && (output_dtype == DataType::UINT32 || output_dtype == DataType::INT32 ||
                                               output_dtype == DataType::FLOAT32 || output_dtype == DataType::BFLOAT16);
}
}  // namespace

Tensor clone(
    const Tensor& input,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    if (dtype.has_value() and clone_cannot_convert_dtype(input.dtype(), *dtype)) {
        return ttnn::typecast(input, *dtype, memory_config);
    }
    return ttnn::prim::clone(input, dtype, memory_config, compute_kernel_config);
}
}  // namespace ttnn
