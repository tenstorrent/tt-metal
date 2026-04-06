// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"

namespace ttnn {

std::vector<Tensor> addcmul_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& tensor1,
    const Tensor& tensor2,
    float value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

std::vector<Tensor> addcdiv_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& tensor1,
    const Tensor& tensor2,
    float value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

std::vector<std::optional<Tensor>> where_bw(
    const Tensor& grad_tensor,
    const Tensor& condition,
    const Tensor& input,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
    std::optional<Tensor> input_grad = std::nullopt,
    std::optional<Tensor> other_grad = std::nullopt);

std::vector<Tensor> lerp_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& end,
    const Tensor& weight,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> lerp_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& end,
    float weight,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

}  // namespace ttnn
