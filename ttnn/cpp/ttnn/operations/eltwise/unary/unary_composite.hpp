// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/validation.hpp"

namespace ttnn {

namespace operations {

namespace unary {

// re-implement tt_eager composite unary op => ttnn composite unary ops.
Tensor deg2rad(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return ttnn::multiply(queue_id, input_tensor, (float)(M_PI / 180.0), std::nullopt, memory_config.value_or(input_tensor.memory_config()), optional_output_tensor);
}
Tensor rad2deg(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return ttnn::multiply(queue_id, input_tensor, (float)(180.0 / M_PI), std::nullopt, memory_config.value_or(input_tensor.memory_config()), optional_output_tensor);
}


// TODO: update these composite unary ops pending decision on TensorAsync implementation.

// TODO: implement these composite unary ops with optional output tensor and queue id.
Tensor acosh(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::acosh(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}

Tensor asinh(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::asinh(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}

Tensor atanh(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::atanh(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}

Tensor cbrt(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::cbrt(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor cosh(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::cosh(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor digamma(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::digamma(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor hardswish(
    uint8_t queue_id,
    const Tensor& input_tensor,
    float scale,
    float shift,
    const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::hardswish(input_tensor, scale, shift, memory_config.value_or(input_tensor.memory_config()));
}
Tensor hardsigmoid(
    uint8_t queue_id,
    const Tensor& input_tensor,
    float scale,
    float shift,
    const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::hardsigmoid(input_tensor, scale, shift, memory_config.value_or(input_tensor.memory_config()));
}
Tensor hardtanh(
    uint8_t queue_id,
    const Tensor& input_tensor,
    float low /* = -1.0f */,
    float high /* = +1.0f */,
    const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::hardtanh(input_tensor, low, high, memory_config.value_or(input_tensor.memory_config()));
}
Tensor lgamma(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::lgamma(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor log1p(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::log1p(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor mish(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::mish(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor multigammaln(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::multigammaln(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor sinh(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::sinh(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor softsign(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::softsign(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor swish(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::swish(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor tanhshrink(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::tanhshrink(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor tril(
    uint8_t queue_id,
    const Tensor& input_tensor,
    int32_t diag=0,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::tril(input_tensor, diag, memory_config.value_or(input_tensor.memory_config()));
}
Tensor triu(
    uint8_t queue_id,
    const Tensor& input_tensor,
    int32_t diag=0,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::triu(input_tensor, diag, memory_config.value_or(input_tensor.memory_config()));
}

}  // namespace unary
}  // namespace operations

// auto prelu = ttnn::leaky_relu;  // Alias for leaky_relu. TODO(#8544): implement PReLU properly

// Other unaries
constexpr auto deg2rad = ttnn::register_operation("ttnn::deg2rad", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::deg2rad));
constexpr auto rad2deg = ttnn::register_operation("ttnn::rad2deg", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::rad2deg));

constexpr auto acosh = ttnn::register_operation("ttnn::acosh", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::acosh));
constexpr auto asinh = ttnn::register_operation("ttnn::asinh", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::asinh));
constexpr auto atanh = ttnn::register_operation("ttnn::atanh", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::atanh));
constexpr auto cbrt = ttnn::register_operation("ttnn::cbrt", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::cbrt));
constexpr auto cosh = ttnn::register_operation("ttnn::cosh", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::cosh));
constexpr auto digamma = ttnn::register_operation("ttnn::digamma", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::digamma));
constexpr auto hardswish = ttnn::register_operation("ttnn::hardswish", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::hardswish));
constexpr auto hardsigmoid =
    ttnn::register_operation("ttnn::hardsigmoid", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::hardsigmoid));
constexpr auto hardtanh = ttnn::register_operation("ttnn::hardtanh", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::hardtanh));
constexpr auto lgamma = ttnn::register_operation("ttnn::lgamma", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::lgamma));
constexpr auto log1p = ttnn::register_operation("ttnn::log1p", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::log1p));
constexpr auto mish = ttnn::register_operation("ttnn::mish", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::mish));
constexpr auto multigammaln =
    ttnn::register_operation("ttnn::multigammaln", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::multigammaln));
constexpr auto sinh = ttnn::register_operation("ttnn::sinh", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::sinh));
constexpr auto softsign = ttnn::register_operation("ttnn::softsign", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::softsign));
constexpr auto swish = ttnn::register_operation("ttnn::swish", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::swish));
constexpr auto tanhshrink =
    ttnn::register_operation("ttnn::tanhshrink", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::tanhshrink));
constexpr auto tril = ttnn::register_operation("ttnn::tril", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::tril));
constexpr auto triu = ttnn::register_operation("ttnn::triu", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::triu));

}  // namespace ttnn
