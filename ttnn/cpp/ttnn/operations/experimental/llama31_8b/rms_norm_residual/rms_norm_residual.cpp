// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/rms_norm_residual/rms_norm_residual.hpp"

#include "ttnn/operations/experimental/llama31_8b/rms_norm_residual/device/rms_norm_residual_device_operation.hpp"

namespace ttnn::experimental::llama31_8b {

std::tuple<ttnn::Tensor, ttnn::Tensor> rms_norm_residual(
    const ttnn::Tensor& input,
    const ttnn::Tensor& residual,
    const ttnn::Tensor& weight,
    float epsilon,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
    const auto config = init_device_compute_kernel_config(
        input.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, false, true, true);
    auto residual_out = create_device_tensor(
        tt::tt_metal::TensorSpec(
            input.logical_shape(),
            tt::tt_metal::TensorLayout(
                DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG)),
        input.device());
    auto output = ttnn::prim::llama31_8b_rms_norm_residual(input, residual, weight, residual_out, epsilon, config);
    return {output, residual_out};
}

bool rms_norm_residual_supported(const ttnn::Tensor& input, const ttnn::Tensor& residual) {
    return rms_norm_residual_fits(input, residual);
}

}  // namespace ttnn::experimental::llama31_8b
