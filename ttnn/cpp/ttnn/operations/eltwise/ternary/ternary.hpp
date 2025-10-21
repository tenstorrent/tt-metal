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
        const std::variant<float, Tensor>& value_true,
        const std::variant<float, Tensor>& value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt);
};

}  // namespace ternary
}  // namespace operations

// Register the where operation
constexpr auto where = ttnn::register_operation<"ttnn::where", operations::ternary::WhereOperation>();

}  // namespace ttnn
