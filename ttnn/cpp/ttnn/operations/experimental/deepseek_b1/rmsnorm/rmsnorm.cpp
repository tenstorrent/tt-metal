// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm.hpp"
#include "device/rmsnorm_device_operation.hpp"
#include <algorithm>

namespace ttnn::operations::experimental::deepseek_b1::rmsnorm {

ttnn::Tensor RmsnormOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& output_tensor,
    float epsilon,
    uint32_t numel,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::rmsnorm(input_tensor, gamma_tensor, output_tensor, epsilon, numel, compute_kernel_config);
}

}  // namespace ttnn::operations::experimental::deepseek_b1::rmsnorm
