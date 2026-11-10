// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "clone.hpp"

#include "device/clone_device_operation.hpp"

namespace ttnn {

Tensor clone(
    const Tensor& input,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::prim::clone(input, dtype, memory_config, compute_kernel_config, sub_core_grids);
}
}  // namespace ttnn
