// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bernoulli_new.hpp"

#include "device/bernoulli_new_device_operation.hpp"

namespace ttnn::operations::bernoulli_new {
Tensor BernoulliNew::invoke(
    const Tensor& input,
    const uint32_t seed,
    const std::optional<Tensor>& output,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::bernoulli_new(input, seed, output, dtype, memory_config, compute_kernel_config);
}
}  // namespace ttnn::operations::bernoulli_new
