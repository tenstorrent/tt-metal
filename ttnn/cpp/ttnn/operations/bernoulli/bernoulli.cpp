
// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bernoulli.hpp"

#include "device/bernoulli_device_operation.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn {

Tensor bernoulli(
    const Tensor& input,
    const uint32_t seed,
    const std::optional<Tensor>& output,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_OP_SCOPE("ttnn::bernoulli");
    return ttnn::prim::bernoulli(input, seed, output, dtype, memory_config, compute_kernel_config);
}

}  // namespace ttnn
