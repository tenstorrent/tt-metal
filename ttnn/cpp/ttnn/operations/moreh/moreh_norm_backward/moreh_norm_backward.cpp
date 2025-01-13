// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm_backward.hpp"

#include "device/moreh_norm_backward_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_norm_backward {
Tensor MorehNormBackward::invoke(
    const Tensor& input,
    const Tensor& output,
    const Tensor& output_grad,
    float p,
    std::optional<std::variant<int64_t, ttnn::SmallVector<int64_t>>> dim,
    bool keepdim,
    const std::optional<Tensor>& input_grad,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::moreh_norm_backward(
        input, output, output_grad, p, dim, keepdim, input_grad, memory_config, compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_norm_backward
