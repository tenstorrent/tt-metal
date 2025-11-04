// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/ternary/common/ternary_op_types.hpp"

namespace ttnn {

namespace operations {

namespace ternary {

// Where Operation
struct WhereOperation {
    static Tensor invoke(
        const Tensor& predicate,
        const TensorScalarVariant& value_true,
        const TensorScalarVariant& value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt);

    template <typename T>
        requires std::same_as<T, int32_t> || std::same_as<T, uint32_t>
    static Tensor invoke(
        const Tensor& predicate,
        const T& value_true,
        const T& value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt);
};

}  // namespace ternary
}  // namespace operations

// Register the where operation
constexpr auto where = ttnn::register_operation<"ttnn::where", operations::ternary::WhereOperation>();

}  // namespace ttnn
