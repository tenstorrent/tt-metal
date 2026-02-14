// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn {
namespace operations::transformer {}  // namespace operations::transformer

namespace transformer {

/**
 * @brief Windowed scaled dot product attention. 
 * This is similar to the standard SDPA but instead of accepting an explicit attention mask, 
 * it accepts cumulative window sequence lengths and builds the attention mask internally 
 * to create block-diagonal attention patterns.
 */
ttnn::Tensor windowed_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const ttnn::Tensor& cu_window_seqlens,
    std::optional<float> scale = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<SDPAProgramConfig> program_config = std::nullopt,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace transformer

}  // namespace ttnn
