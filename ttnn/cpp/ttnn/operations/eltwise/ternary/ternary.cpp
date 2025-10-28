// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary.hpp"

#include <utility>
#include <variant>

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "device/ternary_device_operation.hpp"
#include "device/ternary_op_utils.hpp"
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
    const std::optional<Tensor>& output) {
    log_info(tt::LogOp, "Where Legacy");
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

// Helper function to determine output dtype
inline std::optional<DataType> determine_output_dtype(
    const std::optional<Tensor>& output, const std::optional<DataType>& default_dtype) {
    return output.has_value() ? std::optional<DataType>(output->dtype()) : default_dtype;
}

// Helper function to determine memory config
inline MemoryConfig determine_memory_config(
    const std::optional<MemoryConfig>& memory_config, const MemoryConfig& default_config) {
    return memory_config.value_or(default_config);
}

}  // namespace ternary_utils

namespace {

inline bool is_sharded(const Tensor& t) { return t.memory_config().is_sharded(); }
inline bool is_sharded(const std::optional<MemoryConfig>& mc) { return mc.has_value() && mc->is_sharded(); }
inline bool is_sharded(const std::optional<Tensor>& t) { return t.has_value() && t->memory_config().is_sharded(); }
inline bool is_invalid_bcast(const ttnn::operations::ternary::TernaryBroadcastType& broadcast_type) {
    return broadcast_type == ttnn::operations::ternary::TernaryBroadcastType::INVALID_BCAST;
}

// TTT: tensor, tensor, tensor
Tensor invoke_impl(
    const Tensor& predicate,
    const Tensor& t_true,
    const Tensor& t_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    Tensor condition = predicate;
    auto broadcast_type = ttnn::operations::ternary::get_broadcast_type(
        condition.logical_shape(), t_true.logical_shape(), t_false.logical_shape());
    bool typecast_needed = ternary_utils::typecast_predicate(predicate, t_true, t_false);
    if (typecast_needed) {
        condition = ttnn::typecast(predicate, t_true.dtype());
    }
    if (is_sharded(condition) || is_sharded(t_true) || is_sharded(t_false) || is_sharded(memory_config) ||
        is_sharded(output) || is_invalid_bcast(broadcast_type)) {
        return ternary_utils::where_impl(
            condition,
            t_true,
            t_false,
            ternary_utils::determine_memory_config(memory_config, condition.memory_config()),
            output);
    }
    std::optional<DataType> output_dtype = ternary_utils::determine_output_dtype(output, t_true.dtype());

    log_info(tt::LogOp, "Where LLK - TTT");
    return ttnn::prim::ternary(
        TernaryOpType::WHERE,
        std::move(condition),
        t_true,
        t_false,
        output_dtype,
        ternary_utils::determine_memory_config(memory_config, t_true.memory_config()),
        output);
}

// TTS: tensor, tensor, scalar
Tensor invoke_impl(
    const Tensor& predicate,
    const Tensor& t_true,
    float scalar_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    Tensor condition = predicate;
    auto broadcast_type =
        ttnn::operations::ternary::get_broadcast_type(condition.logical_shape(), t_true.logical_shape());
    bool typecast_needed = ternary_utils::typecast_predicate(predicate, t_true);
    if (typecast_needed) {
        condition = ttnn::typecast(predicate, t_true.dtype());
    }

    if (is_sharded(condition) || is_sharded(t_true) || is_sharded(memory_config) || is_sharded(output) ||
        is_invalid_bcast(broadcast_type)) {
        return ternary_utils::where_impl(
            condition,
            t_true,
            scalar_false,
            ternary_utils::determine_memory_config(memory_config, condition.memory_config()),
            output);
    }

    std::optional<DataType> output_dtype = ternary_utils::determine_output_dtype(output, t_true.dtype());
    log_info(tt::LogOp, "Where LLK - TTS");
    return ttnn::prim::ternary(
        TernaryOpType::WHERE,
        std::move(condition),
        t_true,
        scalar_false,
        output_dtype,
        ternary_utils::determine_memory_config(memory_config, t_true.memory_config()),
        output);
}

// TST: tensor, scalar, tensor
Tensor invoke_impl(
    const Tensor& predicate,
    float scalar_true,
    const Tensor& t_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    Tensor condition = predicate;
    bool typecast_needed = ternary_utils::typecast_predicate(predicate, t_false);
    if (typecast_needed) {
        condition = ttnn::typecast(predicate, t_false.dtype());
    }
    auto broadcast_type =
        ttnn::operations::ternary::get_broadcast_type(condition.logical_shape(), t_false.logical_shape());
    if (is_sharded(condition) || is_sharded(t_false) || is_sharded(memory_config) || is_sharded(output) ||
        is_invalid_bcast(broadcast_type)) {
        return ternary_utils::where_impl(
            condition,
            scalar_true,
            t_false,
            ternary_utils::determine_memory_config(memory_config, condition.memory_config()),
            output);
    }

    std::optional<DataType> output_dtype = ternary_utils::determine_output_dtype(output, t_false.dtype());
    log_info(tt::LogOp, "Where LLK - TST");
    return ttnn::prim::ternary(
        TernaryOpType::WHERE,
        std::move(condition),
        scalar_true,
        t_false,
        output_dtype,
        ternary_utils::determine_memory_config(memory_config, t_false.memory_config()),
        output);
}

// TSS: tensor, scalar, scalar
Tensor invoke_impl(
    const Tensor& condition,
    float t_true,
    float t_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    log_info(tt::LogOp, "Where LLK - TSS");
    unary::UnaryOpType op_type = unary::UnaryOpType::WHERE_TSS;
    return ttnn::operations::unary::Unary_chain::invoke(
        condition,
        {unary::UnaryWithParam{op_type, {static_cast<float>(t_true), static_cast<float>(t_false)}}},
        memory_config,
        output);
}

}  // namespace

Tensor WhereOperation::invoke(
    const Tensor& predicate,
    const TensorScalarVariant& value_true,
    const TensorScalarVariant& value_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    return std::visit(
        [&](const auto& true_val, const auto& false_val) {
            return invoke_impl(predicate, true_val, false_val, memory_config, output);
        },
        value_true,
        value_false);
}

}  // namespace ternary
}  // namespace operations
}  // namespace ttnn
