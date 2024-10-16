
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bernoulli.hpp"

#include "device/bernoulli_device_operation.hpp"

namespace ttnn::operations::bernoulli {
Tensor Bernoulli::invoke(
    const Tensor& input,
    const std::optional<Tensor>& out,
    const std::optional<DataType>& out_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::bernoulli(input, out, out_dtype, memory_config, compute_kernel_config);
}
}  // namespace ttnn::operations::bernoulli
