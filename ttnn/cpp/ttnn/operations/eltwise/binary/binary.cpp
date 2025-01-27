
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary.hpp"

#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/copy.hpp"

namespace ttnn::operations::binary {

namespace detail {

constexpr bool is_associative(BinaryOpType op) {
    return op == BinaryOpType::ADD || op == BinaryOpType::MUL || op == BinaryOpType::EQ || op == BinaryOpType::NE ||
           op == BinaryOpType::LOGICAL_AND || op == BinaryOpType::LOGICAL_OR || op == BinaryOpType::LOGADDEXP ||
           op == BinaryOpType::LOGADDEXP2 || op == BinaryOpType::LOGICAL_XOR;
}

// Tensor - Scalar
inline Tensor binary_impl(
    uint8_t queue_id,
    BinaryOpType binary_op_type,
    const ttnn::Tensor& lhs,
    const float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt) {
    const auto& output_memory_config =
        output.has_value() ? output->memory_config() : memory_config.value_or(lhs.memory_config());
    auto output_tensor = lhs;
    if (binary_op_type == BinaryOpType::GT) {
        output_tensor = ttnn::gt_unary(queue_id, lhs, rhs, output_memory_config, output);
    } else if (binary_op_type == BinaryOpType::LT) {
        output_tensor = ttnn::lt_unary(queue_id, lhs, rhs, output_memory_config, output);
    } else if (binary_op_type == BinaryOpType::NE) {
        output_tensor = ttnn::ne_unary(queue_id, lhs, rhs, output_memory_config, output);
    } else if (binary_op_type == BinaryOpType::GTE) {
        output_tensor =
            ttnn::gez(queue_id, ttnn::sub_sfpu(queue_id, lhs, rhs, output_memory_config), output_memory_config, output);
    } else if (binary_op_type == BinaryOpType::LTE) {
        output_tensor =
            ttnn::lez(queue_id, ttnn::sub_sfpu(queue_id, lhs, rhs, output_memory_config), output_memory_config, output);
    } else if (binary_op_type == BinaryOpType::EQ) {
        output_tensor =
            ttnn::eqz(queue_id, ttnn::sub_sfpu(queue_id, lhs, rhs, output_memory_config), output_memory_config, output);
    } else {
        TT_THROW("Unsupported operation");
    }
    if (dtype.has_value()) {
        output_tensor = ttnn::typecast(queue_id, output_tensor, dtype.value(), std::nullopt, output);
    }
    return output_tensor;
}

// Scalar - Tensor
inline Tensor binary_impl(
    uint8_t queue_id,
    BinaryOpType binary_op_type,
    const float lhs,
    const ttnn::Tensor& rhs,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt) {
    const auto& output_memory_config =
        output.has_value() ? output->memory_config() : memory_config.value_or(rhs.memory_config());
    auto output_tensor = rhs;
    if (binary_op_type == BinaryOpType::GTE) {
        output_tensor =
            ttnn::gez(queue_id, ttnn::sub_sfpu(queue_id, lhs, rhs, output_memory_config), output_memory_config, output);
    } else if (binary_op_type == BinaryOpType::LTE) {
        output_tensor =
            ttnn::lez(queue_id, ttnn::sub_sfpu(queue_id, lhs, rhs, output_memory_config), output_memory_config, output);
    } else if (binary_op_type == BinaryOpType::EQ) {
        output_tensor =
            ttnn::eqz(queue_id, ttnn::sub_sfpu(queue_id, lhs, rhs, output_memory_config), output_memory_config, output);
    } else {
        TT_THROW("Unsupported operation");
    }
    return output_tensor;
}

auto invoke_binary_ng(
    uint8_t queue_id,
    const Tensor& lhs,
    const auto& rhs,
    BinaryOpType binary_op_type,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<unary::FusedActivations>& activations,
    const std::optional<unary::UnaryWithParam>& lhs_activation) {
    const auto& post_activations = activations.has_value() ? tt::stl::Span<const unary::UnaryWithParam>{*activations}
                                                           : tt::stl::Span<const unary::UnaryWithParam>{};
    const auto& lhs_activations =
        lhs_activation.has_value() ? unary::FusedActivations{*lhs_activation} : unary::FusedActivations{};
    const auto& rhs_activations = tt::stl::Span<const unary::UnaryWithParam>{};

    return ttnn::prim::binary_ng(
        queue_id,
        lhs,
        rhs,
        binary_op_type,
        output_dtype,
        memory_config,
        output,
        lhs_activations,
        rhs_activations,
        post_activations);
}

}  // namespace detail

template <BinaryOpType binary_op_type>
Tensor BinaryOperation<binary_op_type>::invoke(
    uint8_t queue_id,
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<unary::FusedActivations>& activations,
    const std::optional<unary::UnaryWithParam>& lhs_activation) {
    return detail::invoke_binary_ng(
        queue_id, lhs, rhs, binary_op_type, output_dtype, memory_config, output, activations, lhs_activation);
}

template <BinaryOpType binary_op_type>
Tensor BinaryOperation<binary_op_type>::invoke(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<unary::FusedActivations>& activations,
    const std::optional<unary::UnaryWithParam>& input_tensor_a_activation) {
    return invoke(
        DefaultQueueId, lhs, rhs, output_dtype, memory_config, output, activations, input_tensor_a_activation);
}

template <BinaryOpType binary_op_type>
Tensor BinaryOperation<binary_op_type>::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<unary::FusedActivations>& activations,
    const std::optional<unary::UnaryWithParam>& lhs_activation) {
    return detail::invoke_binary_ng(
        queue_id, lhs, rhs, binary_op_type, output_dtype, memory_config, output, activations, lhs_activation);
}

// TODO: this case should use BinaryWithScalarProgramConfig and there should be a custom kernel to run this
// Currently, this is exactly how tt::tt_metal::add_unary works
template <BinaryOpType binary_op_type>
Tensor BinaryOperation<binary_op_type>::invoke(
    const ttnn::Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<unary::FusedActivations>& activations,
    const std::optional<unary::UnaryWithParam>& lhs_activation) {
    return BinaryOperation::invoke(
        DefaultQueueId, lhs, rhs, output_dtype, memory_config, output, activations, lhs_activation);
}

template <BinaryOpType binary_op_type>
Tensor RelationalBinary<binary_op_type>::invoke(
    uint8_t queue_id,
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<unary::FusedActivations>& activations,
    const std::optional<unary::UnaryWithParam>& lhs_activation) {
    if (output_dtype.has_value() && output.has_value()) {
        TT_FATAL(
            *output_dtype == output->get_dtype(), "If both output dtype and output tensor provided dtype should match");
    }

    auto output_memory_config = memory_config.value_or(lhs.memory_config());
    DataType dtype = output_dtype.value_or(lhs.get_dtype());
    if (output.has_value()) {
        dtype = output->get_dtype();
    }

    return detail::invoke_binary_ng(
        queue_id, lhs, rhs, binary_op_type, dtype, output_memory_config, output, activations, lhs_activation);
}

template <BinaryOpType binary_op_type>
Tensor RelationalBinary<binary_op_type>::invoke(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<unary::FusedActivations>& activations,
    const std::optional<unary::UnaryWithParam>& lhs_activation) {
    return invoke(DefaultQueueId, lhs, rhs, output_dtype, memory_config, output, activations, lhs_activation);
}

template <BinaryOpType binary_op_type>
Tensor RelationalBinary<binary_op_type>::invoke(
    const ttnn::Tensor& lhs,
    const float rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<unary::FusedActivations>& activations,
    const std::optional<unary::UnaryWithParam>& lhs_activation) {
    return detail::binary_impl(DefaultQueueId, binary_op_type, lhs, rhs, dtype, memory_config, output);
}

template <BinaryOpType binary_op_type>
Tensor RelationalBinary<binary_op_type>::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& lhs,
    const float rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<unary::FusedActivations>& activations,
    const std::optional<unary::UnaryWithParam>& lhs_activation) {
    return detail::binary_impl(DefaultQueueId, binary_op_type, lhs, rhs, dtype, memory_config, output);
}
// scalar - tensor combination not available on Pytorch for this op
template <BinaryOpType binary_op_type>
Tensor RelationalBinary<binary_op_type>::invoke(
    uint8_t queue_id,
    const float lhs,
    const ttnn::Tensor& rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    return detail::binary_impl(DefaultQueueId, binary_op_type, lhs, rhs, memory_config, output);
}

template <BinaryOpType binary_op_type>
Tensor InplaceRelationalBinary<binary_op_type>::invoke(const Tensor& lhs, const Tensor& rhs) {
    return RelationalBinary<binary_op_type>::invoke(
        lhs, rhs, std::nullopt, std::nullopt, lhs, std::nullopt, std::nullopt);
}

template <BinaryOpType binary_op_type>
Tensor InplaceRelationalBinary<binary_op_type>::invoke(const ttnn::Tensor& lhs, const float rhs) {
    return RelationalBinary<binary_op_type>::invoke(
        lhs, rhs, std::nullopt, std::nullopt, lhs, std::nullopt, std::nullopt);
}

template <BinaryOpType binary_op_type>
Tensor InplaceLogicalBinary<binary_op_type>::invoke(const Tensor& lhs, const Tensor& rhs) {
    return BinaryOperation<binary_op_type>::invoke(
        lhs, rhs, std::nullopt, std::nullopt, lhs, std::nullopt, std::nullopt);
}

template <BinaryOpType binary_op_type>
Tensor InplaceBinaryOperation<binary_op_type>::invoke(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<unary::FusedActivations>& activations,
    const std::optional<unary::UnaryWithParam>& lhs_activation) {
    return BinaryOperation<binary_op_type>::invoke(
        lhs, rhs, std::nullopt, std::nullopt, lhs, activations, lhs_activation);
}

template <BinaryOpType binary_op_type>
Tensor InplaceBinaryOperation<binary_op_type>::invoke(
    const ttnn::Tensor& lhs,
    const float rhs,
    const std::optional<unary::FusedActivations>& activations,
    const std::optional<unary::UnaryWithParam>& lhs_activation) {
    return BinaryOperation<binary_op_type>::invoke(
        lhs, rhs, std::nullopt, std::nullopt, lhs, activations, lhs_activation);
}

template <BinaryOpType binary_op_type>
Tensor BinaryOperationSfpu<binary_op_type>::invoke(
    uint8_t queue_id,
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<unary::FusedActivations>& activations,
    const std::optional<unary::UnaryWithParam>& lhs_activation) {
    const auto& output_memory_config = memory_config.value_or(lhs.memory_config());
    const auto dtype = output.has_value() ? output->get_dtype() : output_dtype.value_or(lhs.get_dtype());

    return detail::invoke_binary_ng(
        queue_id, lhs, rhs, binary_op_type, dtype, output_memory_config, output, activations, lhs_activation);
}

template <BinaryOpType binary_op_type>
Tensor BinaryOperationSfpu<binary_op_type>::invoke(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<unary::FusedActivations>& activations,
    const std::optional<unary::UnaryWithParam>& lhs_activation) {
    return invoke(DefaultQueueId, lhs, rhs, output_dtype, memory_config, output, activations, lhs_activation);
}

template struct BinaryOperation<BinaryOpType::ADD>;
template struct InplaceBinaryOperation<BinaryOpType::ADD>;
template struct BinaryOperation<BinaryOpType::SUB>;
template struct InplaceBinaryOperation<BinaryOpType::SUB>;
template struct BinaryOperation<BinaryOpType::MUL>;
template struct InplaceBinaryOperation<BinaryOpType::MUL>;
template struct BinaryOperation<BinaryOpType::LOGICAL_AND>;
template struct BinaryOperation<BinaryOpType::LOGICAL_OR>;
template struct BinaryOperation<BinaryOpType::LOGICAL_XOR>;
template struct BinaryOperation<BinaryOpType::LDEXP>;
template struct BinaryOperation<BinaryOpType::LOGADDEXP>;
template struct BinaryOperation<BinaryOpType::LOGADDEXP2>;
template struct BinaryOperation<BinaryOpType::SQUARED_DIFFERENCE>;
template struct BinaryOperation<BinaryOpType::DIV>;
template struct BinaryOperation<BinaryOpType::BIAS_GELU>;
template struct BinaryOperation<BinaryOpType::RSUB>;

template struct RelationalBinary<BinaryOpType::EQ>;
template struct RelationalBinary<BinaryOpType::NE>;
template struct RelationalBinary<BinaryOpType::GTE>;
template struct RelationalBinary<BinaryOpType::GT>;
template struct RelationalBinary<BinaryOpType::LTE>;
template struct RelationalBinary<BinaryOpType::LT>;

template struct InplaceRelationalBinary<BinaryOpType::GT>;
template struct InplaceRelationalBinary<BinaryOpType::LT>;
template struct InplaceRelationalBinary<BinaryOpType::GTE>;
template struct InplaceRelationalBinary<BinaryOpType::LTE>;
template struct InplaceRelationalBinary<BinaryOpType::EQ>;
template struct InplaceRelationalBinary<BinaryOpType::NE>;

template struct InplaceLogicalBinary<BinaryOpType::LOGICAL_AND>;
template struct InplaceLogicalBinary<BinaryOpType::LOGICAL_OR>;
template struct InplaceLogicalBinary<BinaryOpType::LOGICAL_XOR>;

template struct BinaryOperationSfpu<BinaryOpType::POWER>;
template struct BinaryOperationSfpu<BinaryOpType::BITWISE_AND>;
template struct BinaryOperationSfpu<BinaryOpType::BITWISE_XOR>;
template struct BinaryOperationSfpu<BinaryOpType::BITWISE_OR>;
template struct BinaryOperationSfpu<BinaryOpType::LEFT_SHIFT>;
template struct BinaryOperationSfpu<BinaryOpType::RIGHT_SHIFT>;

}  // namespace ttnn::operations::binary
