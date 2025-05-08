// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_linear.hpp"

#include "ttnn/operations/moreh/moreh_matmul/moreh_matmul.hpp"
namespace ttnn::operations::moreh::moreh_linear {
Tensor MorehLinear::invoke(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::moreh_matmul(input, weight, false, true, output, bias, memory_config, compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_linear
