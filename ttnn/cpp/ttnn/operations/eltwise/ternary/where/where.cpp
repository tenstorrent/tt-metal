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

// Broadcast support utilities
// Local broadcast type with additional UNSUPPORTED value
enum class LocalWhereBroadcastType {
    NONE,        // all tensors have equal shapes
    COL_BCAST,   // column broadcast supported (e.g., [1,1,32,32] and [1,1,32,1])
    UNSUPPORTED  // other broadcast types not yet supported
};

LocalWhereBroadcastType get_where_broadcast_type(
    const Tensor& predicate, const Tensor& value_true, const Tensor& value_false) {
    const auto& pred_shape = predicate.logical_shape();
    const auto& true_shape = value_true.logical_shape();
    const auto& false_shape = value_false.logical_shape();

    // Check if all shapes are the same (no broadcast needed)
    if (pred_shape == true_shape && pred_shape == false_shape) {
        return LocalWhereBroadcastType::NONE;
    }

    // For now, only support column broadcast where all tensors have same rank
    if (pred_shape.rank() != true_shape.rank() || pred_shape.rank() != false_shape.rank()) {
        return LocalWhereBroadcastType::UNSUPPORTED;
    }

    // Check for column broadcast: one of the tensors has width 1, others have same width > 1
    // Example: [1,1,32,32] and [1,1,32,1] -> column broadcast
    bool has_col_bcast = false;

    // Get the last dimension (width) and second-to-last dimension (height)
    auto rank = pred_shape.rank();
    if (rank >= 2) {
        auto pred_w = pred_shape[-1];
        auto pred_h = pred_shape[-2];
        auto true_w = true_shape[-1];
        auto true_h = true_shape[-2];
        auto false_w = false_shape[-1];
        auto false_h = false_shape[-2];

        // Check if all heights match
        if (pred_h == true_h && pred_h == false_h) {
            // Check for column broadcast pattern
            if ((pred_w == true_w && (false_w == 1 || false_w == pred_w)) ||
                (pred_w == false_w && (true_w == 1 || true_w == pred_w)) ||
                (true_w == false_w && (pred_w == 1 || pred_w == true_w))) {
                // Ensure at least one tensor has width 1 and others have the same width > 1
                bool has_width_1 = (pred_w == 1) || (true_w == 1) || (false_w == 1);
                bool others_match = (pred_w == true_w) || (pred_w == false_w) || (true_w == false_w);

                if (has_width_1 && others_match) {
                    // Check that all other dimensions match
                    bool other_dims_match = true;
                    for (int i = 0; i < rank - 2; ++i) {
                        if (pred_shape[i] != true_shape[i] || pred_shape[i] != false_shape[i]) {
                            other_dims_match = false;
                            break;
                        }
                    }
                    if (other_dims_match) {
                        has_col_bcast = true;
                    }
                }
            }
        }
    }

    return has_col_bcast ? LocalWhereBroadcastType::COL_BCAST : LocalWhereBroadcastType::UNSUPPORTED;
}

bool can_use_llk_with_broadcast(const Tensor& predicate, const Tensor& value_true, const Tensor& value_false) {
    auto broadcast_type = get_where_broadcast_type(predicate, value_true, value_false);
    return broadcast_type == LocalWhereBroadcastType::NONE || broadcast_type == LocalWhereBroadcastType::COL_BCAST;
}

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
            if (ternary_utils::can_use_llk_with_broadcast(predicate, t_true, t_false)) {
                auto broadcast_type = ternary_utils::get_where_broadcast_type(predicate, t_true, t_false);
                if (broadcast_type == ternary_utils::LocalWhereBroadcastType::NONE) {
                    log_info(tt::LogOp, "Where LLK - TTT (same shape)");
                } else if (broadcast_type == ternary_utils::LocalWhereBroadcastType::COL_BCAST) {
                    log_info(tt::LogOp, "Where LLK - TTT (column broadcast)");
                }
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
                log_info(tt::LogOp, "Where LLK - TTS (same shape)");
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
                log_info(tt::LogOp, "Where LLK - TST (same shape)");
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
            log_info(tt::LogOp, "Where LLK - TSS");
            std::optional<DataType> output_dtype = output.has_value() ? std::optional<DataType>(output->dtype())
                                                                      : std::optional<DataType>(predicate.dtype());
            unary::UnaryOpType op_type = unary::UnaryOpType::WHERE_TSS;

            return ttnn::operations::unary::Unary_chain::invoke(
                queue_id,
                predicate,
                {unary::UnaryWithParam{op_type, {static_cast<float>(t_true), static_cast<float>(t_false)}}},
                memory_config,
                output);
        }
    }

    log_info(tt::LogOp, "Where - legacy");
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
