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

    // Check if we can use fast ternary LLK path (TTT, TTS, TST cases)
    if (!has_shard_spec) {
        if (is_value_true_Tensor && is_value_false_Tensor) {
            // TTT case: tensor-tensor-tensor
            const auto& t_true = std::get<Tensor>(value_true);
            const auto& t_false = std::get<Tensor>(value_false);
            if (ternary_utils::have_same_shape(t_true, predicate) &&
                ternary_utils::have_same_shape(predicate, t_false)) {
                log_debug(tt::LogOp, "Where LLK - TTT");
                std::optional<DataType> output_dtype = output.has_value() ? std::optional<DataType>(output->dtype())
                                                                          : std::optional<DataType>(predicate.dtype());
                return ttnn::prim::where(
                    queue_id,
                    predicate,
                    t_true,
                    t_false,
                    output_dtype,
                    memory_config.value_or(predicate.memory_config()),
                    output);
            }
        } else if (is_value_true_Tensor && !is_value_false_Tensor) {
            // TTS case: tensor-tensor-scalar
            const auto& t_true = std::get<Tensor>(value_true);
            if (ternary_utils::have_same_shape(t_true, predicate)) {
                log_debug(tt::LogOp, "Where LLK - TTS");
                float scalar_false = std::get<float>(value_false);
                std::optional<DataType> output_dtype = output.has_value() ? std::optional<DataType>(output->dtype())
                                                                          : std::optional<DataType>(predicate.dtype());
                return ttnn::prim::where(
                    queue_id,
                    predicate,
                    t_true,
                    scalar_false,
                    output_dtype,
                    memory_config.value_or(predicate.memory_config()),
                    output);
            }
        } else if (!is_value_true_Tensor && is_value_false_Tensor) {
            // TST case: tensor-scalar-tensor
            const auto& t_false = std::get<Tensor>(value_false);
            if (ternary_utils::have_same_shape(predicate, t_false)) {
                log_debug(tt::LogOp, "Where LLK - TST");
                float scalar_true = std::get<float>(value_true);
                std::optional<DataType> output_dtype = output.has_value() ? std::optional<DataType>(output->dtype())
                                                                          : std::optional<DataType>(predicate.dtype());
                return ttnn::prim::where(
                    queue_id,
                    predicate,
                    scalar_true,
                    t_false,
                    output_dtype,
                    memory_config.value_or(predicate.memory_config()),
                    output);
            }
        } else if (!is_value_true_Tensor && !is_value_false_Tensor && !has_shard_spec) {
            // TSS case: tensor-scalar-scalar
            const auto& t_true = std::get<float>(value_true);
            const auto& t_false = std::get<float>(value_false);
            log_debug(tt::LogOp, "Where LLK - TSS");
            unary::UnaryOpType op_type = unary::UnaryOpType::WHERE_TSS;

            return ttnn::operations::unary::Unary_chain::invoke(
                queue_id,
                predicate,
                {unary::UnaryWithParam{op_type, {static_cast<float>(t_true), static_cast<float>(t_false)}}},
                memory_config,
                output);
        }
    }

    log_debug(tt::LogOp, "Where - legacy");
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
