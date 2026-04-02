// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot.hpp"

#include "ttnn/operations/moreh/moreh_dot/device/moreh_dot_device_operation.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn {

Tensor moreh_dot(
    const Tensor& input_a,
    const Tensor& input_b,
    const std::optional<Tensor>& output,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    ttnn::graph::ScopedCompositeTrace _trace("ttnn::moreh_dot");
    return ttnn::prim::moreh_dot(input_a, input_b, output, dtype, memory_config, compute_kernel_config);
}

}  // namespace ttnn
