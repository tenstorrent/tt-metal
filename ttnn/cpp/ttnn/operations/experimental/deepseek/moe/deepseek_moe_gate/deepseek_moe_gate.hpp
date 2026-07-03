// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::deepseek::moe {

// DeepSeek V3 MoE gate (top-8 routing + score normalization) on height-sharded tensors.
// Fills preallocated output_tensor and output_indices_tensor in place.
std::tuple<tt::tt_metal::Tensor, tt::tt_metal::Tensor> deepseek_moe_gate(
    const tt::tt_metal::Tensor& input_tensor,
    const tt::tt_metal::Tensor& bias_tensor,
    const tt::tt_metal::Tensor& input_indices_tensor,
    const tt::tt_metal::Tensor& output_tensor,
    const tt::tt_metal::Tensor& output_indices_tensor,
    float eps,
    float scaling_factor,
    bool enable_sigmoid);

}  // namespace ttnn::experimental::deepseek::moe
