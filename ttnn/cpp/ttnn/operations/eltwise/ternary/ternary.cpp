// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary.hpp"

#include <utility>
#include <variant>

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "device/ternary_device_operation.hpp"
#include "device/ternary_op_utils.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ternary_composite_op.hpp"

namespace ttnn::operations::ternary {

namespace ternary_utils {

// where - ternary operator y = (predicate) ? value_true : value_false; elementwise
// y = (predicate >= 0)*value_true + (predicate < 0)*value_false
Tensor where_impl(
    const Tensor& predicate,
    const auto& value_true,
    const auto& value_false,
    const MemoryConfig& memory_config,
    const std::optional<Tensor>& output) {
    log_debug(tt::LogOp, "Where Legacy");
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

// Helper function to determine memory config for broadcast operations
// This properly adjusts the shard spec when the output shape differs from the input tensor's shape
inline MemoryConfig determine_memory_config_for_broadcast(
    const std::optional<MemoryConfig>& memory_config, const Tensor& sharded_tensor, const ttnn::Shape& other_shape) {
    if (memory_config.has_value()) {
        return *memory_config;
    }
    // If the sharded tensor's memory config is sharded, adjust the shard spec for the broadcast output shape
    if (sharded_tensor.memory_config().is_sharded()) {
        return ttnn::operations::binary_ng::compute_mem_config_actual(sharded_tensor, other_shape);
    }
    return sharded_tensor.memory_config();
}

}  // namespace ternary_utils

namespace {

// Functions will be enabled in future when porting more ops to the ternary infra
// inline bool is_sharded(const Tensor& t) { return t.memory_config().is_sharded(); }
// inline bool is_sharded(const std::optional<MemoryConfig>& mc) { return mc.has_value() && mc->is_sharded(); }
// inline bool is_sharded(const std::optional<Tensor>& t) { return t.has_value() && t->memory_config().is_sharded(); }
inline bool is_invalid_bcast(const ttnn::operations::ternary::TernaryBroadcastType& broadcast_type) {
    return broadcast_type == ttnn::operations::ternary::TernaryBroadcastType::INVALID_BCAST;
}

// TTT: tensor, tensor, tensor
Tensor invoke_impl(
    const Tensor& predicate,
    const Tensor& t_true,
    const Tensor& t_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    Tensor condition = predicate;
    auto broadcast_type = ttnn::operations::ternary::get_broadcast_type(
        condition.logical_shape(), t_true.logical_shape(), t_false.logical_shape());
    bool typecast_needed = ternary_utils::typecast_predicate(predicate, t_true, t_false);
    if (typecast_needed) {
        condition = ttnn::typecast(predicate, t_true.dtype());
    }

    if (is_invalid_bcast(broadcast_type)) {
        return ternary_utils::where_impl(
            condition,
            t_true,
            t_false,
            ternary_utils::determine_memory_config(memory_config, predicate.memory_config()),
            output);
    }

    std::optional<DataType> output_dtype = ternary_utils::determine_output_dtype(output, t_true.dtype());
    log_debug(tt::LogOp, "Where LLK - TTT");
    return ttnn::prim::ternary(
        TernaryOpType::WHERE,
        condition,
        t_true,
        t_false,
        output_dtype,
        ternary_utils::determine_memory_config(memory_config, t_true.memory_config()),
        output,
        sub_core_grids);
}

// TTS: tensor, tensor, scalar
Tensor invoke_impl(
    const Tensor& predicate,
    const Tensor& t_true,
    float scalar_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    Tensor condition = predicate;
    bool typecast_needed = ternary_utils::typecast_predicate(predicate, t_true);
    if (typecast_needed) {
        condition = ttnn::typecast(predicate, t_true.dtype());
    }

    return binary::WhereOperationWithScalar<binary::BinaryOpType::WHERE_TTS>::invoke(
        condition,
        t_true,
        scalar_false,
        ternary_utils::determine_memory_config_for_broadcast(memory_config, t_true, condition.logical_shape()),
        output,
        sub_core_grids);
}

// TST: tensor, scalar, tensor
Tensor invoke_impl(
    const Tensor& predicate,
    float scalar_true,
    const Tensor& t_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    Tensor condition = predicate;
    bool typecast_needed = ternary_utils::typecast_predicate(predicate, t_false);
    if (typecast_needed) {
        condition = ttnn::typecast(predicate, t_false.dtype());
    }

    return binary::WhereOperationWithScalar<binary::BinaryOpType::WHERE_TST>::invoke(
        condition,
        t_false,
        scalar_true,
        ternary_utils::determine_memory_config_for_broadcast(memory_config, t_false, condition.logical_shape()),
        output,
        sub_core_grids);
}

// TSS: tensor, scalar, scalar
Tensor invoke_impl(
    const Tensor& condition,
    float t_true,
    float t_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    log_debug(tt::LogOp, "Where LLK - TSS");
    //  TODO: add sub_core_grids functionality to Unary Infra
    if (sub_core_grids.has_value()) {
        TT_THROW("Subcore grids are not supported for WhereOperation TSS variant");
    }
    return ttnn::where_tss(condition, t_true, t_false, memory_config, output);
}

}  // namespace

Tensor WhereOperation::invoke(
    const Tensor& predicate,
    const TensorScalarVariant& value_true,
    const TensorScalarVariant& value_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return std::visit(
        [&](const auto& true_val, const auto& false_val) {
            return invoke_impl(predicate, true_val, false_val, memory_config, output, sub_core_grids);
        },
        value_true,
        value_false);
}

template <typename T>
    requires std::same_as<T, int32_t> || std::same_as<T, uint32_t>
Tensor WhereOperation::invoke(
    const Tensor& predicate,
    const T& value_true,
    const T& value_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    if (sub_core_grids.has_value()) {
        TT_THROW("Subcore grids are not supported for WhereOperation TSS variant");
    }
    return ttnn::where_tss(predicate, value_true, value_false, memory_config, output);
}

template Tensor WhereOperation::invoke<int32_t>(
    const Tensor&,
    const int32_t&,
    const int32_t&,
    const std::optional<MemoryConfig>&,
    const std::optional<Tensor>&,
    const std::optional<CoreRangeSet>&);
template Tensor WhereOperation::invoke<uint32_t>(
    const Tensor&,
    const uint32_t&,
    const uint32_t&,
    const std::optional<MemoryConfig>&,
    const std::optional<Tensor>&,
    const std::optional<CoreRangeSet>&);

Tensor AddcmulOperation::invoke(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    log_debug(tt::LogOp, "Addcmul LLK - TTT");

    // Only TTT variant is supported for addcmul
    auto broadcast_type = ttnn::operations::ternary::get_broadcast_type(
        input_a.logical_shape(), input_b.logical_shape(), input_c.logical_shape());

    bool is_any_input_block_format =
        is_block_float(input_a.dtype()) || is_block_float(input_b.dtype()) || is_block_float(input_c.dtype());
    bool is_subtile_bcast = (broadcast_type == TernaryBroadcastType::ROW_BCAST) ||
                            (broadcast_type == TernaryBroadcastType::COL_BCAST) ||
                            (broadcast_type == TernaryBroadcastType::SCALAR_BCAST);
    bool is_input_int32 = (input_a.dtype() == DataType::INT32) && (input_b.dtype() == DataType::INT32) &&
                          (input_c.dtype() == DataType::INT32);

    if (is_invalid_bcast(broadcast_type) || (is_any_input_block_format && is_subtile_bcast) || is_input_int32) {
        log_debug(tt::LogOp, "Addcmul Fallback - TTT");
        // Fall back to composite implementation for unsupported cases
        // For block-format ROW bcast of ttnn.mul, legacy binary bcast implementation is used.
        // For int32 inputs, composite implementation is used.
        return _addcmul(input_a, input_b, input_c, value, memory_config);
    }

    // Use LLK implementation - pass value as scalar parameter
    log_debug(tt::LogOp, "Addcmul LLK - TTT");
    return ttnn::prim::ternary(
        TernaryOpType::ADDCMUL,
        input_a,
        input_b,
        input_c,
        value,
        ternary_utils::determine_output_dtype(output, input_a.dtype()),
        ternary_utils::determine_memory_config(memory_config, input_a.memory_config()),
        output,
        std::nullopt);
}

Tensor AddcdivOperation::invoke(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    log_debug(tt::LogOp, "Addcdiv LLK - TTT");

    // Only TTT variant is supported for addcdiv
    auto broadcast_type = ttnn::operations::ternary::get_broadcast_type(
        input_a.logical_shape(), input_b.logical_shape(), input_c.logical_shape());

    bool is_any_input_block_format =
        is_block_float(input_a.dtype()) || is_block_float(input_b.dtype()) || is_block_float(input_c.dtype());
    bool is_input_int32 = (input_a.dtype() == DataType::INT32) && (input_b.dtype() == DataType::INT32) &&
                          (input_c.dtype() == DataType::INT32);

    TT_FATAL(!is_input_int32, "Addcdiv TTT does not support INT32 inputs.");

    if (is_invalid_bcast(broadcast_type) || is_any_input_block_format) {
        log_debug(tt::LogOp, "Addcdiv Fallback - TTT");
        // Fall back to composite implementation for unsupported cases
        return _addcdiv(input_a, input_b, input_c, value, memory_config);
    }

    // Use LLK implementation - pass value as scalar parameter
    log_debug(tt::LogOp, "Addcdiv LLK - TTT");
    return ttnn::prim::ternary(
        TernaryOpType::ADDCDIV,
        input_a,
        input_b,
        input_c,
        value,
        ternary_utils::determine_output_dtype(output, input_a.dtype()),
        ternary_utils::determine_memory_config(memory_config, input_a.memory_config()),
        output,
        std::nullopt);
}

Tensor LerpOperation::invoke(
    const Tensor& input,
    const Tensor& end,
    float weight,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    log_debug(tt::LogOp, "Lerp LLK - TTS (scalar weight)");

    auto broadcast_type = ttnn::operations::ternary::get_broadcast_type(input.logical_shape(), end.logical_shape());

    bool is_any_input_block_format = is_block_float(input.dtype()) || is_block_float(end.dtype());
    bool is_subtile_bcast = (broadcast_type == TernaryBroadcastType::ROW_BCAST) ||
                            (broadcast_type == TernaryBroadcastType::COL_BCAST) ||
                            (broadcast_type == TernaryBroadcastType::SCALAR_BCAST);
    bool is_input_int32 = (input.dtype() == DataType::INT32) && (end.dtype() == DataType::INT32);

    if (is_invalid_bcast(broadcast_type) || (is_any_input_block_format && is_subtile_bcast) || is_input_int32) {
        log_debug(tt::LogOp, "Lerp Fallback - TTS");
        return _lerp_overload(input, end, weight, memory_config);
    }

    log_debug(tt::LogOp, "Lerp LLK - TTS");
    return ttnn::prim::ternary(
        TernaryOpType::LERP,
        input,
        end,
        weight,
        ternary_utils::determine_output_dtype(output, input.dtype()),
        ternary_utils::determine_memory_config(memory_config, input.memory_config()),
        output,
        std::nullopt);
}

Tensor LerpOperation::invoke(
    const Tensor& input,
    const Tensor& end,
    const Tensor& weight,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    log_debug(tt::LogOp, "Lerp LLK - TTT (tensor weight)");

    auto broadcast_type = ttnn::operations::ternary::get_broadcast_type(
        input.logical_shape(), end.logical_shape(), weight.logical_shape());

    bool is_any_input_block_format =
        is_block_float(input.dtype()) || is_block_float(end.dtype()) || is_block_float(weight.dtype());
    bool is_subtile_bcast = (broadcast_type == TernaryBroadcastType::ROW_BCAST) ||
                            (broadcast_type == TernaryBroadcastType::COL_BCAST) ||
                            (broadcast_type == TernaryBroadcastType::SCALAR_BCAST);
    bool is_input_int32 =
        (input.dtype() == DataType::INT32) && (end.dtype() == DataType::INT32) && (weight.dtype() == DataType::INT32);

    if (is_invalid_bcast(broadcast_type) || (is_any_input_block_format && is_subtile_bcast) || is_input_int32) {
        log_debug(tt::LogOp, "Lerp Fallback - TTT");
        return _lerp(input, end, weight, memory_config);
    }

    log_debug(tt::LogOp, "Lerp LLK - TTT");
    return ttnn::prim::ternary(
        TernaryOpType::LERP,
        input,
        end,
        weight,
        ternary_utils::determine_output_dtype(output, input.dtype()),
        ternary_utils::determine_memory_config(memory_config, input.memory_config()),
        output,
        std::nullopt);
}

}  // namespace ttnn::operations::ternary
