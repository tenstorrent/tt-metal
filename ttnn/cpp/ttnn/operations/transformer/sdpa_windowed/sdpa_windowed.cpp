// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_windowed.hpp"

#include <utility>

#include "ttnn/operations/transformer/sdpa_windowed/device/sdpa_windowed_device_operation.hpp"
#include "ttnn/common/queue_id.hpp"

namespace ttnn::operations::transformer {

ttnn::Tensor ExecuteWindowedScaledDotProductAttention::invoke(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& cu_window_seqlens,
    std::optional<float> scale,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<SDPAProgramConfig> program_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

    return ttnn::prim::windowed_scaled_dot_product_attention(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        cu_window_seqlens,
        scale,
        memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG),
        std::move(program_config),
        kernel_config_val);
}

}  // namespace ttnn::operations::transformer
