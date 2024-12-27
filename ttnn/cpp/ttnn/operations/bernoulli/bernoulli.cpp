
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bernoulli.hpp"

#include "device/bernoulli_device_operation.hpp"

namespace ttnn::operations::bernoulli {
Tensor Bernoulli::invoke(
    const Tensor& input,
    const uint32_t seed,
    const std::optional<Tensor>& output,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::bernoulli(input, seed, output, dtype, memory_config, compute_kernel_config);
}
}  // namespace ttnn::operations::bernoulli
