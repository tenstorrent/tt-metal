// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where.hpp"

#include <utility>
#include <variant>

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "device/where_device_operation.hpp"
#include "device/where_utils.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttnn {
namespace operations {
namespace ternary {

namespace ternary_utils {

// where - ternary operator y = (predicate) ? value_true : value_false; elementwise
// y = (predicate >= 0)*value_true + (predicate < 0)*value_false

Tensor where_impl(
    const Tensor& predicate,
    const auto& value_true,
    const auto& value_false,
    const MemoryConfig& memory_config,
    std::optional<Tensor> output) {
    using FusedActivations = tt::stl::Span<const unary::EltwiseUnaryWithParam>;
    constexpr auto dtype = std::nullopt;
    const auto get_multiplied = [&](const Tensor& condition, const auto& value) -> Tensor {
        return ttnn::multiply(
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
        get_multiplied(ttnn::gtz(predicate, memory_config), value_true),
        get_multiplied(ttnn::lez(predicate, memory_config), value_false),
        dtype,
        memory_config,
        output,
        /* post_activations */ FusedActivations{},
        /* lhs_activations */ FusedActivations{},
        /* rhs_activations */ FusedActivations{},
        /* use_legacy */ false);
}

inline bool have_same_shape(const Tensor& a, const Tensor& b) { return (a.logical_shape() == b.logical_shape()); }

inline bool typecast_predicate(const Tensor& predicate, const Tensor& t_true, const Tensor& t_false) {
    return !is_floating_point(predicate.dtype()) && is_floating_point(t_true.dtype()) &&
           is_floating_point(t_false.dtype());
}

inline bool typecast_predicate(const Tensor& predicate, const Tensor& b) {
    return !is_floating_point(predicate.dtype()) && is_floating_point(b.dtype());
}

}  // namespace ternary_utils

// Helper function to check if sharding is present
bool has_sharding(
    const Tensor& predicate,
    const std::variant<float, Tensor>& value_true,
    const std::variant<float, Tensor>& value_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    auto check_tensor_sharding = [](const auto& tensor) {
        return std::holds_alternative<Tensor>(tensor) && std::get<Tensor>(tensor).memory_config().is_sharded();
    };

    return predicate.memory_config().is_sharded() || check_tensor_sharding(value_true) ||
           check_tensor_sharding(value_false) || (memory_config.has_value() && memory_config->is_sharded()) ||
           (output.has_value() && output->memory_config().is_sharded());
}

// Helper function to handle typecasting and prim::where invocation for TTT case
Tensor handle_ttt_case(
    Tensor condition,
    const Tensor& t_true,
    const Tensor& t_false,
    const std::optional<DataType>& output_dtype,
    const MemoryConfig& memory_config,
    std::optional<Tensor> output) {
    bool typecast_needed = ternary_utils::typecast_predicate(condition, t_true, t_false);
    if (typecast_needed) {
        condition = ttnn::typecast(condition, t_true.dtype());
    }

    auto broadcast_type = ttnn::operations::ternary::get_broadcast_type(
        condition.logical_shape(), t_true.logical_shape(), t_false.logical_shape());

    if (broadcast_type != ttnn::operations::ternary::WhereBroadcastType::INVALID_BCAST) {
        log_info(tt::LogOp, "Where LLK - TTT");
        return ttnn::prim::where(condition, t_true, t_false, output_dtype, memory_config, output);
    }
    return Tensor();  // Invalid tensor to indicate fallback needed
}

// Helper function to handle TTS case
Tensor handle_tts_case(
    Tensor condition,
    const Tensor& t_true,
    float scalar_false,
    const std::optional<DataType>& output_dtype,
    const MemoryConfig& memory_config,
    std::optional<Tensor> output) {
    bool typecast_needed = ternary_utils::typecast_predicate(condition, t_true);
    if (typecast_needed) {
        condition = ttnn::typecast(condition, t_true.dtype());
    }

    auto broadcast_type =
        ttnn::operations::ternary::get_broadcast_type(condition.logical_shape(), t_true.logical_shape());

    if (broadcast_type != ttnn::operations::ternary::WhereBroadcastType::INVALID_BCAST) {
        log_info(tt::LogOp, "Where LLK - TTS");
        return ttnn::prim::where(condition, t_true, scalar_false, output_dtype, memory_config, output);
    }
    return Tensor();  // Invalid tensor to indicate fallback needed
}

// Helper function to handle TST case
Tensor handle_tst_case(
    Tensor condition,
    float scalar_true,
    const Tensor& t_false,
    const std::optional<DataType>& output_dtype,
    const MemoryConfig& memory_config,
    std::optional<Tensor> output) {
    bool typecast_needed = ternary_utils::typecast_predicate(condition, t_false);
    if (typecast_needed) {
        condition = ttnn::typecast(condition, t_false.dtype());
    }

    auto broadcast_type =
        ttnn::operations::ternary::get_broadcast_type(condition.logical_shape(), t_false.logical_shape());

    if (broadcast_type != ttnn::operations::ternary::WhereBroadcastType::INVALID_BCAST) {
        log_info(tt::LogOp, "Where LLK - TST");
        return ttnn::prim::where(condition, scalar_true, t_false, output_dtype, memory_config, output);
    }
    return Tensor();  // Invalid tensor to indicate fallback needed
}

// Helper function to handle TSS case
Tensor handle_tss_case(
    const Tensor& predicate,
    float t_true,
    float t_false,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> output) {
    log_info(tt::LogOp, "Where LLK - TSS");
    unary::UnaryOpType op_type = unary::UnaryOpType::WHERE_TSS;
    return ttnn::operations::unary::Unary_chain::invoke(
        predicate,
        {unary::UnaryWithParam{op_type, {static_cast<float>(t_true), static_cast<float>(t_false)}}},
        memory_config,
        output);
}

Tensor WhereOperation::invoke(
    const Tensor& predicate,
    const std::variant<float, Tensor>& value_true,
    const std::variant<float, Tensor>& value_false,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> output) {
    bool is_value_true_tensor = std::holds_alternative<Tensor>(value_true);
    bool is_value_false_tensor = std::holds_alternative<Tensor>(value_false);
    Tensor condition = predicate;

    // Check if sharding prevents optimized path
    if (has_sharding(predicate, value_true, value_false, memory_config, output)) {
        log_info(tt::LogOp, "Where - legacy (sharding detected)");
        return std::visit(
            [&](const auto&... values) {
                return ternary_utils::where_impl(
                    condition, values..., memory_config.value_or(predicate.memory_config()), std::move(output));
            },
            value_true,
            value_false);
    }

    // Try optimized paths in order of preference
    if (is_value_true_tensor && is_value_false_tensor) {
        // TTT case: tensor-tensor-tensor
        const auto& t_true = std::get<Tensor>(value_true);
        const auto& t_false = std::get<Tensor>(value_false);
        std::optional<DataType> output_dtype =
            output.has_value() ? std::optional<DataType>(output->dtype()) : std::optional<DataType>(t_true.dtype());
        Tensor result = handle_ttt_case(
            condition, t_true, t_false, output_dtype, memory_config.value_or(t_true.memory_config()), output);
        if (result.is_allocated()) {
            return result;
        }
    } else if (is_value_true_tensor && !is_value_false_tensor) {
        // TTS case: tensor-tensor-scalar
        const auto& t_true = std::get<Tensor>(value_true);
        float scalar_false = std::get<float>(value_false);
        std::optional<DataType> output_dtype =
            output.has_value() ? std::optional<DataType>(output->dtype()) : std::optional<DataType>(t_true.dtype());
        Tensor result = handle_tts_case(
            condition, t_true, scalar_false, output_dtype, memory_config.value_or(t_true.memory_config()), output);
        if (result.is_allocated()) {
            return result;
        }
    } else if (!is_value_true_tensor && is_value_false_tensor) {
        // TST case: tensor-scalar-tensor
        float scalar_true = std::get<float>(value_true);
        const auto& t_false = std::get<Tensor>(value_false);
        std::optional<DataType> output_dtype =
            output.has_value() ? std::optional<DataType>(output->dtype()) : std::optional<DataType>(t_false.dtype());
        Tensor result = handle_tst_case(
            condition, scalar_true, t_false, output_dtype, memory_config.value_or(t_false.memory_config()), output);
        if (result.is_allocated()) {
            return result;
        }
    } else if (!is_value_true_tensor && !is_value_false_tensor) {
        // TSS case: tensor-scalar-scalar
        float t_true = std::get<float>(value_true);
        float t_false = std::get<float>(value_false);
        return handle_tss_case(predicate, t_true, t_false, memory_config, output);
    }

    // Fallback to legacy implementation
    log_info(tt::LogOp, "Where - legacy");
    return std::visit(
        [&](const auto&... values) {
            return ternary_utils::where_impl(
                condition, values..., memory_config.value_or(predicate.memory_config()), std::move(output));
        },
        value_true,
        value_false);
}

}  // namespace ternary
}  // namespace operations
}  // namespace ttnn
