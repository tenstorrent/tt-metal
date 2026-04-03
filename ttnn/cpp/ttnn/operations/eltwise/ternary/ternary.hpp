// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <optional>
#include <variant>
#include <reflect>

#include "ttnn/operations/eltwise/ternary/common/ternary_op_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

Tensor where(
    const Tensor& predicate,
    const operations::ternary::TensorScalarVariant& value_true,
    const operations::ternary::TensorScalarVariant& value_false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

template <typename T>
    requires std::same_as<T, int32_t> || std::same_as<T, uint32_t>
Tensor where(
    const Tensor& predicate,
    const T& value_true,
    const T& value_false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor addcmul(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    operations::ternary::ScalarVariant value = 1.0f,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);

Tensor addcdiv(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value = 1.0f,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);

Tensor lerp(
    const Tensor& input,
    const Tensor& end,
    float weight,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);

Tensor lerp(
    const Tensor& input,
    const Tensor& end,
    const Tensor& weight,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);

}  // namespace ttnn
