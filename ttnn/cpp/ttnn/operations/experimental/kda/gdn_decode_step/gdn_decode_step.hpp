// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

// One gated-delta-rule decode step (B = 1, or 1 <= B <= 32 with conv_hist / conv_taps); `state` is updated in place.
// Shapes and the packed conv layout are documented in the Python docstring (gdn_decode_step_nanobind.cpp).
ttnn::Tensor gdn_decode_step(
    const ttnn::Tensor& qkv,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& g,
    const ttnn::Tensor& state,
    const ttnn::Tensor& weight,
    uint32_t num_value_heads,
    uint32_t num_key_heads,
    uint32_t key_dim,
    uint32_t value_dim,
    std::optional<float> scale = std::nullopt,
    float l2_epsilon = 1e-6f,
    float norm_epsilon = 1e-6f,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    ttnn::DataType output_dtype = ttnn::DataType::BFLOAT16,
    const std::optional<ttnn::Tensor>& conv_hist = std::nullopt,
    const std::optional<ttnn::Tensor>& conv_taps = std::nullopt,
    uint32_t qkvz_dim = 0,
    bool fast_mode = false,
    bool fp32_state = false,
    const std::optional<ttnn::Tensor>& row_pos = std::nullopt);

}  // namespace ttnn::experimental::kda
