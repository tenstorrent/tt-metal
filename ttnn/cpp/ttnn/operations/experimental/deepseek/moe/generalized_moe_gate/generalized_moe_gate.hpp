// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::deepseek::moe {

// Generalized (ungrouped) MoE gate routing + score normalization on height-sharded tensors. With
// grouped=true it instead runs the DeepSeek grouped gate (8 groups × 32 -> top-8). Fills the preallocated
// output_tensor and output_indices_tensor in place.
std::tuple<ttnn::Tensor, ttnn::Tensor> generalized_moe_gate(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& bias_tensor,
    const ttnn::Tensor& input_indices_tensor,
    const ttnn::Tensor& output_tensor,
    const ttnn::Tensor& output_indices_tensor,
    float eps,
    float scaling_factor,
    bool enable_sigmoid,
    uint32_t topk = 8,
    bool output_softmax = false,
    bool grouped = false);

}  // namespace ttnn::experimental::deepseek::moe
