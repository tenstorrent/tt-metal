// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "clone.hpp"

#include "device/clone_device_operation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttnn {

Tensor clone(
    const Tensor& input,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    // The clone kernel converts through copy_tile with differing unpack/pack formats, which is only correct between
    // float formats. Integer <-> float conversions need the typecast SFPU path.
    if (dtype.has_value() && is_int_float_conversion(input.dtype(), dtype.value())) {
        return ttnn::typecast(input, dtype.value(), memory_config.value_or(input.memory_config()));
    }
    return ttnn::prim::clone(input, dtype, memory_config, compute_kernel_config);
}
}  // namespace ttnn
