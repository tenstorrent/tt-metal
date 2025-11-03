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

// Thread-local variable for addcmul value
inline thread_local float addcmul_value = 1.0f;

// Where Operation
struct WhereOperation {
    static Tensor invoke(
        const Tensor& predicate,
        const TensorScalarVariant& value_true,
        const TensorScalarVariant& value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt);
};

// Addcmul Operation
struct AddcmulOperation {
    static Tensor invoke(
        const Tensor& input_a,
        const Tensor& input_b,
        const Tensor& input_c,
        float value = 1.0f,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt);
};

}  // namespace ternary
}  // namespace operations

// Register the operations
constexpr auto where = ttnn::register_operation<"ttnn::where", operations::ternary::WhereOperation>();
constexpr auto addcmul = ttnn::register_operation<"ttnn::addcmul", operations::ternary::AddcmulOperation>();

}  // namespace ttnn
