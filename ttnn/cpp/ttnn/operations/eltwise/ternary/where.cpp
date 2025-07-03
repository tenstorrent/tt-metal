// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "where.hpp"

#include <utility>
#include <variant>

#include "ttnn/common/queue_id.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn {
namespace operations {
namespace ternary {

namespace ternary_utils {

// where - ternary operator y = (predicate) ? value_true : value_false; elementwise
// y = (predicate >= 0)*value_true + (predicate < 0)*value_false

Tensor where_impl(
    QueueId queue_id,
    const Tensor& predicate,
    const auto& value_true,
    const auto& value_false,
    const MemoryConfig& memory_config,
    std::optional<Tensor> output) {
    using FusedActivations = tt::stl::Span<const unary::UnaryWithParam>;
    constexpr auto dtype = std::nullopt;
    const auto get_multiplied = [&](const Tensor& condition, const auto& value) -> Tensor {
        return ttnn::multiply(
            queue_id,
            condition,
            value,
            dtype,
            memory_config,
            /* output */ std::nullopt,
            /* post_activations */ FusedActivations{},
            /* lhs_activations */ FusedActivations{},
            /* rhs_activations */ FusedActivations{},
            /* use_legacy */ false);
    };

    return ttnn::add(
        queue_id,
        get_multiplied(ttnn::gtz(queue_id, predicate, memory_config), value_true),
        get_multiplied(ttnn::lez(queue_id, predicate, memory_config), value_false),
        dtype,
        memory_config,
        output,
        /* post_activations */ FusedActivations{},
        /* lhs_activations */ FusedActivations{},
        /* rhs_activations */ FusedActivations{},
        /* use_legacy */ false);
}

}  // namespace ternary_utils
Tensor WhereOperation::invoke(
    QueueId queue_id,
    const Tensor& predicate,
    const std::variant<float, Tensor>& value_true,
    const std::variant<float, Tensor>& value_false,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> output) {
    return std::visit(
        [&](const auto&... values) {
            return ternary_utils::where_impl(
                queue_id, predicate, values..., memory_config.value_or(predicate.memory_config()), std::move(output));
        },
        value_true,
        value_false);
}

}  // namespace ternary
}  // namespace operations
}  // namespace ttnn
