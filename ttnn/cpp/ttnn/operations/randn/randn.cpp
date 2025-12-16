// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "randn.hpp"

#include "ttnn/operations/randn/device/randn_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::randn {

Tensor Randn::invoke(
    const ttnn::Shape& shape,
    MeshDevice& device,
    const DataType dtype,
    const Layout layout,
    const MemoryConfig& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    uint32_t seed) {
    auto tensor = ttnn::prim::randn(shape, dtype, layout, memory_config, device, compute_kernel_config, seed);
    if (layout != Layout::TILE) {
        tensor = ttnn::to_layout(tensor, layout);
    }
    return tensor;
}
}  // namespace ttnn::operations::randn
