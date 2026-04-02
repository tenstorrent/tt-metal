
// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "uniform.hpp"

#include "device/uniform_device_operation.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn {

Tensor uniform(
    const Tensor& input,
    const float from,
    const float to,
    const uint32_t seed,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    ttnn::graph::ScopedCompositeTrace _trace("ttnn::uniform");
    return ttnn::prim::uniform(input, from, to, seed, memory_config, compute_kernel_config);
}

}  // namespace ttnn
