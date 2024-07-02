// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental//tt_dnn/op_library/composite/composite_ops.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core.hpp"

namespace ttnn {

namespace operations {

namespace ternary {

Tensor where(
    const Tensor& predicate,
    const Tensor& true_value,
    const Tensor& false_value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    auto original_shape = predicate.get_shape();
    auto predicate_4d = ttnn::unsqueeze_to_4D(predicate);
    auto true_value_4d = ttnn::unsqueeze_to_4D(true_value);
    auto false_value_4d = ttnn::unsqueeze_to_4D(false_value);
    auto output = tt::tt_metal::where(
        predicate_4d, true_value_4d, false_value_4d, memory_config.value_or(predicate.memory_config()));
    return ttnn::reshape(output, original_shape);
}

Tensor where(
    const Tensor& predicate,
    const float true_value,
    const Tensor& false_value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    auto original_shape = predicate.get_shape();
    auto predicate_4d = ttnn::unsqueeze_to_4D(predicate);
    auto false_value_4d = ttnn::unsqueeze_to_4D(false_value);
    auto output = tt::tt_metal::where(
        predicate_4d, true_value, false_value_4d, memory_config.value_or(predicate.memory_config()));
    return ttnn::reshape(output, original_shape);
}

Tensor where(
    const Tensor& predicate,
    const Tensor& true_value,
    const float false_value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    auto original_shape = predicate.get_shape();
    auto predicate_4d = ttnn::unsqueeze_to_4D(predicate);
    auto true_value_4d = ttnn::unsqueeze_to_4D(true_value);
    auto output = tt::tt_metal::where(
        predicate_4d, true_value_4d, false_value, memory_config.value_or(predicate.memory_config()));
    return ttnn::reshape(output, original_shape);
}

Tensor where(
    const Tensor& predicate,
    const float true_value,
    const float false_value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    auto original_shape = predicate.get_shape();
    auto predicate_4d = ttnn::unsqueeze_to_4D(predicate);
    auto output =
        tt::tt_metal::where(predicate_4d, true_value, false_value, memory_config.value_or(predicate.memory_config()));
    return ttnn::reshape(output, original_shape);
}

}  // namespace ternary
}  // namespace operations

constexpr auto where = ttnn::register_operation("ttnn::where", TO_LAMBDA(ttnn::operations::ternary::where));

}  // namespace ttnn
