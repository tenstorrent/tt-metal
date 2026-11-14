// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/sdpa_bh/sdpa_bh.hpp"

#include "ttnn/operations/experimental/llama31_8b/sdpa_bh/device/sdpa_bh_device_operation.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::experimental::llama31_8b {

namespace {
DeviceComputeKernelConfig sdpa_kernel_config(const Tensor& q, std::optional<DeviceComputeKernelConfig> cfg) {
    return init_device_compute_kernel_config(
        q.device()->arch(), cfg, tt::tt_metal::MathFidelity::HiFi2, true, false, false);
}
}  // namespace

ttnn::Tensor sdpa_bh(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::operations::transformer::SDPAProgramConfig& program_config,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    bool output_concat_heads) {
    return ttnn::prim::llama31_8b_sdpa_bh(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        scale,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        program_config,
        sdpa_kernel_config(input_tensor_q, compute_kernel_config),
        output_concat_heads);
}

bool sdpa_bh_supported(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    return sdpa_bh_supported(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        program_config,
        sdpa_kernel_config(input_tensor_q, compute_kernel_config));
}

}  // namespace ttnn::experimental::llama31_8b
