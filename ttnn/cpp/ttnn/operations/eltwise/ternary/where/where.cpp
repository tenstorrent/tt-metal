// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where.hpp"

#include <utility>
#include <variant>

#include "ttnn/common/queue_id.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "device/where_device_operation.hpp"

namespace ttnn {
namespace operations {
namespace ternary {

namespace ternary_utils {

// where - ternary operator y = (predicate) ? value_true : value_false; elementwise
// y = (predicate >= 0)*value_true + (predicate < 0)*value_false

template <typename T1, typename T2>
Tensor where_impl(
    QueueId queue_id,
    const Tensor& predicate,
    const T1& value_true,
    const T2& value_false,
    const MemoryConfig& memory_config,
    std::optional<Tensor> output) {
    if constexpr (std::is_same_v<std::decay_t<T1>, Tensor> && std::is_same_v<std::decay_t<T2>, Tensor>) {
        // Both are Tensors: use ttnn::prim::where - also need to add non-bcast check

        std::optional<DataType> output_dtype =
            output.has_value() ? std::optional<DataType>(output->dtype()) : std::optional<DataType>(predicate.dtype());
        return ttnn::prim::where(queue_id, predicate, value_true, value_false, output_dtype, memory_config, output);
    } else {
        // At least one is a float: use the alternate code
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
}

inline bool have_same_shape(const Tensor& a, const Tensor& b) { return (a.logical_shape() == b.logical_shape()); }

}  // namespace ternary_utils
Tensor WhereOperation::invoke(
    QueueId queue_id,
    const Tensor& predicate,
    const std::variant<float, Tensor>& value_true,
    const std::variant<float, Tensor>& value_false,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> output) {
    bool is_value_true_Tensor = std::holds_alternative<Tensor>(value_true);
    bool is_value_false_Tensor = std::holds_alternative<Tensor>(value_false);

    bool has_shard_spec =
        predicate.memory_config().is_sharded() ||
        (is_value_true_Tensor ? std::get<Tensor>(value_true).memory_config().is_sharded() : false) ||
        (is_value_false_Tensor ? std::get<Tensor>(value_false).memory_config().is_sharded() : false) ||
        (memory_config.has_value() ? memory_config.value().is_sharded() : false) ||
        (output.has_value() ? output.value().memory_config().is_sharded() : false);

    // Check if we can use LLK where: TTT, TST, or TTS cases with same shapes and no sharding
    bool can_use_llk_where = false;
    if (!has_shard_spec) {
        if (is_value_true_Tensor && is_value_false_Tensor) {
            // TTT case: both tensors must have same shape as predicate
            const auto& t_true = std::get<Tensor>(value_true);
            const auto& t_false = std::get<Tensor>(value_false);
            bool shapes_match =
                ternary_utils::have_same_shape(t_true, predicate) && ternary_utils::have_same_shape(predicate, t_false);
            can_use_llk_where = shapes_match;
        } else if (is_value_true_Tensor && !is_value_false_Tensor) {
            // TTS case: only value_true tensor must have same shape as predicate
            const auto& t_true = std::get<Tensor>(value_true);
            bool shapes_match = ternary_utils::have_same_shape(t_true, predicate);
            can_use_llk_where = shapes_match;
        } else if (!is_value_true_Tensor && is_value_false_Tensor) {
            // TST case: only value_false tensor must have same shape as predicate
            const auto& t_false = std::get<Tensor>(value_false);
            bool shapes_match = ternary_utils::have_same_shape(predicate, t_false);
            can_use_llk_where = shapes_match;
        }
    }

    if (can_use_llk_where) {
        std::cout << "LLK where op" << std::endl;
        std::optional<DataType> output_dtype =
            output.has_value() ? std::optional<DataType>(output->dtype()) : std::optional<DataType>(predicate.dtype());

        // Call ttnn::prim::where based on the case
        if (is_value_true_Tensor && is_value_false_Tensor) {
            // TTT case
            return ttnn::prim::where(
                queue_id,
                predicate,
                std::get<Tensor>(value_true),
                std::get<Tensor>(value_false),
                output_dtype,
                memory_config.value_or(predicate.memory_config()),
                output);
        } else if (is_value_true_Tensor && !is_value_false_Tensor) {
            // TTS case
            return ttnn::prim::where(
                queue_id,
                predicate,
                std::get<Tensor>(value_true),
                std::get<float>(value_false),
                output_dtype,
                memory_config.value_or(predicate.memory_config()),
                output);
        } else if (!is_value_true_Tensor && is_value_false_Tensor) {
            // TST case
            return ttnn::prim::where(
                queue_id,
                predicate,
                std::get<float>(value_true),
                std::get<Tensor>(value_false),
                output_dtype,
                memory_config.value_or(predicate.memory_config()),
                output);
        }
    }

    std::cout << "Legacy where op" << std::endl;
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
