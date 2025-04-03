// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "clone.hpp"

#include "device/clone_device_operation.hpp"

namespace tt {
namespace tt_metal {
enum class DataType;
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::operations::data_movement::clone {

Tensor Clone::invoke(
    const Tensor& input,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::clone(input, dtype, memory_config, compute_kernel_config);
}
}  // namespace ttnn::operations::data_movement::clone
