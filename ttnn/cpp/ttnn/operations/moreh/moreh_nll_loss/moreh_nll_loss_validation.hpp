// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::nll_loss_validation {

inline void validate_device_tensor(
    const Tensor& tensor, const Tensor& reference_tensor, const std::string_view name, const DataType expected_dtype) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must be allocated in a device buffer", name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{} must use TILE layout", name);
    TT_FATAL(tensor.dtype() == expected_dtype, "{} must have dtype {}, got {}", name, expected_dtype, tensor.dtype());
    TT_FATAL(tensor.device() == reference_tensor.device(), "{} must be on the same device as the input tensor", name);
}

inline void validate_target_shape(const Tensor& input_tensor, const Tensor& target_tensor) {
    const auto& input_shape = input_tensor.logical_shape();
    const auto& target_shape = target_tensor.logical_shape();

    TT_FATAL(input_shape.rank() >= 2, "NLL loss input rank must be at least 2, got {}", input_shape.rank());
    TT_FATAL(
        target_shape.rank() + 1 == input_shape.rank(),
        "NLL loss target rank must be input rank minus one, got input rank {} and target rank {}",
        input_shape.rank(),
        target_shape.rank());

    for (uint32_t target_dim = 0; target_dim < target_shape.rank(); ++target_dim) {
        const uint32_t input_dim = target_dim == 0 ? 0 : target_dim + 1;
        TT_FATAL(
            target_shape[target_dim] == input_shape[input_dim],
            "NLL loss target shape must equal the input shape with class dimension 1 removed; mismatch at target "
            "dimension {} (expected {}, got {})",
            target_dim,
            input_shape[input_dim],
            target_shape[target_dim]);
    }
}

inline void validate_weight(const Tensor& input_tensor, const std::optional<Tensor>& weight_tensor) {
    if (!weight_tensor.has_value()) {
        return;
    }

    validate_device_tensor(*weight_tensor, input_tensor, "weight_tensor", DataType::BFLOAT16);
    TT_FATAL(
        weight_tensor->logical_volume() == input_tensor.logical_shape()[1],
        "NLL loss weight must contain one value per class (expected {}, got {})",
        input_tensor.logical_shape()[1],
        weight_tensor->logical_volume());
}

inline void validate_scalar_tensor(
    const Tensor& input_tensor, const Tensor& scalar_tensor, const std::string_view name) {
    validate_device_tensor(scalar_tensor, input_tensor, name, DataType::BFLOAT16);
    TT_FATAL(scalar_tensor.logical_volume() == 1, "{} must contain exactly one logical element", name);
}

inline void validate_forward(
    const Tensor& input_tensor,
    const Tensor& target_tensor,
    const std::string& reduction,
    const std::optional<Tensor>& weight_tensor,
    const std::optional<Tensor>& divisor_tensor,
    const std::optional<Tensor>& output_tensor) {
    validate_device_tensor(input_tensor, input_tensor, "input_tensor", DataType::BFLOAT16);
    TT_FATAL(
        reduction == "none" || reduction == "sum" || reduction == "mean",
        "NLL loss reduction must be one of 'none', 'sum', or 'mean', got '{}'",
        reduction);
    validate_device_tensor(target_tensor, input_tensor, "target_tensor", DataType::INT32);
    validate_target_shape(input_tensor, target_tensor);
    validate_weight(input_tensor, weight_tensor);

    if (divisor_tensor.has_value()) {
        validate_scalar_tensor(input_tensor, *divisor_tensor, "divisor_tensor");
    }

    if (!output_tensor.has_value()) {
        return;
    }

    validate_device_tensor(*output_tensor, input_tensor, "output_tensor", DataType::BFLOAT16);
    if (reduction != "none") {
        TT_FATAL(output_tensor->logical_volume() == 1, "Reduced NLL loss output must contain one logical element");
        return;
    }

    const auto& output_shape = output_tensor->logical_shape();
    const auto& target_shape = target_tensor.logical_shape();
    const bool exact_shape = output_shape == target_shape;
    const bool rank_two_alias = input_tensor.logical_shape().rank() == 2 && target_shape.rank() == 1 &&
                                output_shape.rank() == 2 && output_shape[0] == 1 && output_shape[1] == target_shape[0];
    TT_FATAL(
        exact_shape || rank_two_alias,
        "Unreduced NLL loss output shape must match the target shape (or [1, N] for a rank-2 input)");
}

inline void validate_backward(
    const Tensor& target_tensor,
    const Tensor& output_grad_tensor,
    const std::optional<Tensor>& weight_tensor,
    const std::optional<Tensor>& input_grad_tensor,
    const std::optional<Tensor>& divisor_tensor) {
    TT_FATAL(input_grad_tensor.has_value(), "NLL loss backward requires input_grad_tensor");
    const Tensor& input_grad = *input_grad_tensor;
    validate_device_tensor(input_grad, input_grad, "input_grad_tensor", DataType::BFLOAT16);
    validate_device_tensor(target_tensor, input_grad, "target_tensor", DataType::INT32);
    validate_target_shape(input_grad, target_tensor);
    validate_scalar_tensor(input_grad, output_grad_tensor, "output_grad_tensor");
    validate_weight(input_grad, weight_tensor);
    if (divisor_tensor.has_value()) {
        validate_scalar_tensor(input_grad, *divisor_tensor, "divisor_tensor");
    }
}

}  // namespace ttnn::operations::moreh::nll_loss_validation
