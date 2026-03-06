
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/complex_binary/device/complex_binary_op.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn {

std::vector<Tensor> atan2_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> xlogy_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> hypot_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> ldexp_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> logaddexp_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> logaddexp2_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> squared_difference_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> min_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> max_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<std::optional<Tensor>> subalpha_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    float alpha,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt,
    const std::optional<Tensor>& other_grad = std::nullopt);

std::vector<std::optional<Tensor>> rsub_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt,
    const std::optional<Tensor>& other_grad = std::nullopt);

std::vector<std::optional<Tensor>> concat_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_a_arg,
    const Tensor& other,
    int dim,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt,
    const std::optional<Tensor>& other_grad = std::nullopt);

std::vector<std::optional<Tensor>> mul_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    float scalar,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt);

std::vector<std::optional<Tensor>> mul_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const Tensor& other_tensor_arg,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt,
    const std::optional<Tensor>& other_grad = std::nullopt);

std::vector<ComplexTensor> mul_bw(
    const ComplexTensor& grad_tensor,
    const ComplexTensor& input_a,
    const ComplexTensor& other,
    const tt::tt_metal::MemoryConfig& output_mem_config);

std::vector<std::optional<Tensor>> assign_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt);

std::vector<std::optional<Tensor>> assign_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    const Tensor& other_tensor,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt,
    const std::optional<Tensor>& other_grad = std::nullopt);

std::vector<Tensor> bias_gelu_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& input_b,
    const std::string& approximate,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> bias_gelu_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float bias,
    const std::string& approximate,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<std::optional<Tensor>> addalpha_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    float alpha,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt,
    const std::optional<Tensor>& other_grad = std::nullopt);

std::vector<std::optional<Tensor>> add_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float alpha,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt);

std::vector<std::optional<Tensor>> add_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt,
    const std::optional<Tensor>& other_grad = std::nullopt);

std::vector<ComplexTensor> add_bw(
    const ComplexTensor& grad_tensor,
    const ComplexTensor& input_a,
    const ComplexTensor& other,
    float alpha,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<std::optional<Tensor>> sub_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float alpha,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt);

std::vector<std::optional<Tensor>> sub_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt,
    const std::optional<Tensor>& other_grad = std::nullopt);

std::vector<ComplexTensor> sub_bw(
    const ComplexTensor& grad_tensor,
    const ComplexTensor& input_a,
    const ComplexTensor& other,
    float alpha,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<std::optional<Tensor>> div_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float scalar,
    const std::optional<std::string>& rounding_mode = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt);

std::vector<std::optional<Tensor>> div_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<std::string>& rounding_mode = std::nullopt,
    const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt,
    const std::optional<Tensor>& other_grad = std::nullopt);

std::vector<ComplexTensor> div_bw(
    const ComplexTensor& grad_tensor,
    const ComplexTensor& input_a,
    const ComplexTensor& other,
    const tt::tt_metal::MemoryConfig& output_mem_config);

std::vector<Tensor> remainder_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float scalar,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> remainder_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> fmod_bw(
    const Tensor& grad_tensor,
    const Tensor& input_tensor,
    float scalar,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

std::vector<Tensor> fmod_bw(
    const Tensor& grad_tensor,
    const Tensor& input_a,
    const Tensor& other,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt);

}  // namespace ttnn
