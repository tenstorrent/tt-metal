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
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    // The clone kernel converts through copy_tile with differing unpack/pack formats, which is only correct between
    // float formats. Integer <-> float conversions need the typecast SFPU path. typecast picks its own
    // fp32_dest_acc_en, so compute_kernel_config does not apply here, and typecast rejects sub_core_grids with a
    // ROW_MAJOR input (clone rejected any ROW_MAJOR dtype change before, so that case was never supported).
    if (dtype.has_value() && is_int_float_conversion(input.dtype(), dtype.value())) {
        return ttnn::typecast(
            input, dtype.value(), memory_config.value_or(input.memory_config()), std::nullopt, sub_core_grids);
    }
    return ttnn::prim::clone(input, dtype, memory_config, compute_kernel_config, sub_core_grids);
}
}  // namespace ttnn
