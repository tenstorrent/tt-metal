// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "device/rmsnorm_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_b1::rmsnorm {

struct RmsnormOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& gamma_tensor,
        const ttnn::Tensor& output_tensor,
        float epsilon = 1e-6,
        uint32_t numel = 0,
        const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::deepseek_b1::rmsnorm

namespace ttnn::prim {
constexpr auto rmsnorm = ttnn::register_operation<
    "ttnn::prim::rmsnorm",
    ttnn::operations::experimental::deepseek_b1::rmsnorm::RmsnormDeviceOperation>();
}  // namespace ttnn::prim

namespace ttnn::experimental::deepseek_b1 {

constexpr auto rmsnorm = ttnn::register_operation<
    "ttnn::experimental::deepseek_b1::rmsnorm",
    ttnn::operations::experimental::deepseek_b1::rmsnorm::RmsnormOperation>();

}  // namespace ttnn::experimental::deepseek_b1
