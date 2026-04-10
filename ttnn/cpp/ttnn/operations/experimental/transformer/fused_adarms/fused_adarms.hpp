// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental {

// Fused adaptive RMSNorm (Pi0.5 style): projects conditioning vector to
// (scale, shift, gate) via a linear projection, then applies RMSNorm with
// the scale and shift, returning (normed, gate).
//
// Equivalent Python:
//   modulation = ttnn.linear(cond, dense_weight, bias=dense_bias)
//   scale, shift, gate = ttnn.chunk(modulation, 3, dim=-1)
//   normed = ttnn.rms_norm(x, weight=scale, bias=shift, epsilon=eps)
//   return normed, gate
//
// Note: dense_bias should have the +1 offset pre-baked into the scale
// portion (first hidden_dim elements), so `scale` from the slice already
// represents (1 + learned_scale).
std::tuple<ttnn::Tensor, ttnn::Tensor> fused_adarms(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dense_weight,
    const ttnn::Tensor& dense_bias,
    const ttnn::Tensor& cond,
    float epsilon = 1e-6f,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
