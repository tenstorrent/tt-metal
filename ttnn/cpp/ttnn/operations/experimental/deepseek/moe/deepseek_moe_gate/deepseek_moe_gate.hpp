// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::deepseek::moe {

// DeepSeek V3 MoE gate (top-8 routing + score normalization) on height-sharded tensors.
// Fills preallocated output_tensor and output_indices_tensor in place.
std::tuple<ttnn::Tensor, ttnn::Tensor> deepseek_moe_gate(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& bias_tensor,
    const ttnn::Tensor& input_indices_tensor,
    const ttnn::Tensor& output_tensor,
    const ttnn::Tensor& output_indices_tensor,
    float eps,
    float scaling_factor,
    bool enable_sigmoid);

}  // namespace ttnn::experimental::deepseek::moe
