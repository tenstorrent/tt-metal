// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "regime_a_matmul.hpp"

#include "device/regime_a_matmul_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental {

ttnn::Tensor regime_a_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::experimental::prim::RegimeAMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::prim::regime_a_matmul(
        input_tensor, weight_tensor, config, memory_config, dtype, compute_kernel_config);
}

}  // namespace ttnn::experimental
