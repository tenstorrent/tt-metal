// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot.hpp"

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/moreh/moreh_dot/device/moreh_dot_device_operation.hpp"

namespace ttnn {

Tensor moreh_dot(
    const Tensor& input_a,
    const Tensor& input_b,
    const std::optional<Tensor>& output,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        !output.has_value() || !dtype.has_value() || output->dtype() == dtype.value(),
        "dtype {} does not match the output tensor dtype {}",
        dtype.value_or(input_a.dtype()),
        output.has_value() ? output->dtype() : input_a.dtype());
    const DataType output_dtype = output.has_value() ? output->dtype() : dtype.value_or(input_a.dtype());
    if (output_dtype == input_a.dtype()) {
        return ttnn::prim::moreh_dot(input_a, input_b, output, dtype, memory_config, compute_kernel_config);
    }
    const Tensor result =
        ttnn::prim::moreh_dot(input_a, input_b, std::nullopt, input_a.dtype(), memory_config, compute_kernel_config);
    return ttnn::typecast(result, output_dtype, memory_config, output);
}

}  // namespace ttnn
