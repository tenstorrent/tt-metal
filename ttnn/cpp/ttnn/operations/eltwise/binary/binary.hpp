
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"

namespace ttnn {

namespace operations {
namespace unary {
struct UnaryWithParam;
}
namespace binary {

bool is_legacy_only(
    const Tensor& lhs,
    const auto& rhs,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations);

template <BinaryOpType binary_op_type>
struct BinaryOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& lhs,
        const Tensor& rhs,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations = {},
        const std::optional<bool>& use_legacy = std::nullopt);

    static Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& lhs,
        float rhs,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations = {},
        const std::optional<bool>& use_legacy = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct RelationalBinary {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& lhs,
        const Tensor& rhs,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations = {},
        const std::optional<bool>& use_legacy = std::nullopt);

    static Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& lhs,
        float rhs,
        const std::optional<const DataType>& dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations = {},
        const std::optional<bool>& use_legacy = std::nullopt);

    // rhs - tensor combination not available on Pytorch for this op
    static Tensor invoke(
        QueueId queue_id,
        float rhs,
        const ttnn::Tensor& lhs,
        const std::optional<const DataType>& dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct InplaceRelationalBinary {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& lhs,
        const Tensor& rhs,
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);

    static Tensor invoke(
        QueueId queue_id,
        const Tensor& lhs,
        float rhs,
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct InplaceLogicalBinary {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& lhs,
        const Tensor& rhs,
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct InplaceBinaryOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& lhs,
        const Tensor& rhs,
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);

    static Tensor invoke(
        QueueId queue_id,
        const Tensor& lhs,
        float rhs,
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct BinaryOperationSfpu {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& lhs,
        const Tensor& rhs,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct BinaryOperationAddalpha {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& lhs,
        const Tensor& rhs,
        float alpha,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct BinaryOperationSubalpha {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& lhs,
        const Tensor& rhs,
        float alpha,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt);
};

}  // namespace binary
}  // namespace operations

constexpr auto add =
    ttnn::register_operation<"ttnn::add", operations::binary::BinaryOperation<operations::binary::BinaryOpType::ADD>>();
constexpr auto add_ = ttnn::register_operation<
    "ttnn::add_",
    operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::ADD>>();
constexpr auto subtract = ttnn::
    register_operation<"ttnn::subtract", operations::binary::BinaryOperation<operations::binary::BinaryOpType::SUB>>();
constexpr auto subtract_ = ttnn::register_operation<
    "ttnn::subtract_",
    operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::SUB>>();
constexpr auto multiply = ttnn::
    register_operation<"ttnn::multiply", operations::binary::BinaryOperation<operations::binary::BinaryOpType::MUL>>();
constexpr auto multiply_ = ttnn::register_operation<
    "ttnn::multiply_",
    operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::MUL>>();
constexpr auto eq =
    ttnn::register_operation<"ttnn::eq", operations::binary::RelationalBinary<operations::binary::BinaryOpType::EQ>>();
constexpr auto ne =
    ttnn::register_operation<"ttnn::ne", operations::binary::RelationalBinary<operations::binary::BinaryOpType::NE>>();
constexpr auto ge =
    ttnn::register_operation<"ttnn::ge", operations::binary::RelationalBinary<operations::binary::BinaryOpType::GE>>();
constexpr auto gt =
    ttnn::register_operation<"ttnn::gt", operations::binary::RelationalBinary<operations::binary::BinaryOpType::GT>>();
constexpr auto le =
    ttnn::register_operation<"ttnn::le", operations::binary::RelationalBinary<operations::binary::BinaryOpType::LE>>();
constexpr auto lt =
    ttnn::register_operation<"ttnn::lt", operations::binary::RelationalBinary<operations::binary::BinaryOpType::LT>>();
constexpr auto logical_and = ttnn::register_operation<
    "ttnn::logical_and",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LOGICAL_AND>>();
constexpr auto logical_or = ttnn::register_operation<
    "ttnn::logical_or",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LOGICAL_OR>>();
constexpr auto logical_xor = ttnn::register_operation<
    "ttnn::logical_xor",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LOGICAL_XOR>>();
constexpr auto ldexp = ttnn::
    register_operation<"ttnn::ldexp", operations::binary::BinaryOperation<operations::binary::BinaryOpType::LDEXP>>();
constexpr auto ldexp_ = ttnn::register_operation<
    "ttnn::ldexp_",
    operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::LDEXP>>();
constexpr auto logaddexp = ttnn::register_operation<
    "ttnn::logaddexp",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LOGADDEXP>>();
constexpr auto logaddexp_ = ttnn::register_operation<
    "ttnn::logaddexp_",
    operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::LOGADDEXP>>();
constexpr auto logaddexp2 = ttnn::register_operation<
    "ttnn::logaddexp2",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LOGADDEXP2>>();
constexpr auto logaddexp2_ = ttnn::register_operation<
    "ttnn::logaddexp2_",
    operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::LOGADDEXP2>>();
constexpr auto squared_difference = ttnn::register_operation<
    "ttnn::squared_difference",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::SQUARED_DIFFERENCE>>();
constexpr auto squared_difference_ = ttnn::register_operation<
    "ttnn::squared_difference_",
    operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::SQUARED_DIFFERENCE>>();
constexpr auto divide = ttnn::
    register_operation<"ttnn::divide", operations::binary::BinaryOperation<operations::binary::BinaryOpType::DIV>>();
constexpr auto divide_ = ttnn::register_operation<
    "ttnn::divide_",
    operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::DIV>>();
constexpr auto gt_ = ttnn::register_operation<
    "ttnn::gt_",
    operations::binary::InplaceRelationalBinary<operations::binary::BinaryOpType::GT>>();
constexpr auto ge_ = ttnn::register_operation<
    "ttnn::ge_",
    operations::binary::InplaceRelationalBinary<operations::binary::BinaryOpType::GE>>();
constexpr auto le_ = ttnn::register_operation<
    "ttnn::le_",
    operations::binary::InplaceRelationalBinary<operations::binary::BinaryOpType::LE>>();
constexpr auto lt_ = ttnn::register_operation<
    "ttnn::lt_",
    operations::binary::InplaceRelationalBinary<operations::binary::BinaryOpType::LT>>();
constexpr auto logical_and_ = ttnn::register_operation<
    "ttnn::logical_and_",
    operations::binary::InplaceLogicalBinary<operations::binary::BinaryOpType::LOGICAL_AND>>();
constexpr auto logical_or_ = ttnn::register_operation<
    "ttnn::logical_or_",
    operations::binary::InplaceLogicalBinary<operations::binary::BinaryOpType::LOGICAL_OR>>();
constexpr auto logical_xor_ = ttnn::register_operation<
    "ttnn::logical_xor_",
    operations::binary::InplaceLogicalBinary<operations::binary::BinaryOpType::LOGICAL_XOR>>();
constexpr auto eq_ = ttnn::register_operation<
    "ttnn::eq_",
    operations::binary::InplaceRelationalBinary<operations::binary::BinaryOpType::EQ>>();
constexpr auto ne_ = ttnn::register_operation<
    "ttnn::ne_",
    operations::binary::InplaceRelationalBinary<operations::binary::BinaryOpType::NE>>();
constexpr auto rsub_ = ttnn::register_operation<
    "ttnn::rsub_",
    operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::RSUB>>();
constexpr auto bias_gelu_ = ttnn::register_operation<
    "ttnn::bias_gelu_",
    operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::BIAS_GELU>>();
constexpr auto addalpha = ttnn::register_operation<
    "ttnn::addalpha",
    operations::binary::BinaryOperationAddalpha<operations::binary::BinaryOpType::ADDALPHA>>();
constexpr auto subalpha = ttnn::register_operation<
    "ttnn::subalpha",
    operations::binary::BinaryOperationSubalpha<operations::binary::BinaryOpType::SUBALPHA>>();
constexpr auto logical_right_shift = ttnn::register_operation<
    "ttnn::logical_right_shift",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LOGICAL_RIGHT_SHIFT>>();
constexpr auto xlogy = ttnn::
    register_operation<"ttnn::xlogy", operations::binary::BinaryOperation<operations::binary::BinaryOpType::XLOGY>>();

template <typename InputBType>
ttnn::Tensor operator+(const ttnn::Tensor& lhs, InputBType rhs) {
    return add(lhs, rhs);
}

template <typename InputBType>
ttnn::Tensor operator-(const ttnn::Tensor& lhs, InputBType rhs) {
    return subtract(lhs, rhs);
}

template <typename InputBType>
ttnn::Tensor operator*(const ttnn::Tensor& lhs, InputBType rhs) {
    return multiply(lhs, rhs);
}

template <typename InputBType>
ttnn::Tensor operator==(const ttnn::Tensor& lhs, InputBType rhs) {
    return eq(lhs, rhs);
}

template <typename InputBType>
ttnn::Tensor operator!=(const ttnn::Tensor& lhs, InputBType rhs) {
    return ne(lhs, rhs);
}

template <typename InputBType>
ttnn::Tensor operator>(const ttnn::Tensor& lhs, InputBType rhs) {
    return gt(lhs, rhs);
}

template <typename InputBType>
ttnn::Tensor operator>=(const ttnn::Tensor& lhs, InputBType rhs) {
    return ge(lhs, rhs);
}

template <typename InputBType>
ttnn::Tensor operator<(const ttnn::Tensor& lhs, InputBType rhs) {
    return lt(lhs, rhs);
}

template <typename InputBType>
ttnn::Tensor operator<=(const ttnn::Tensor& lhs, InputBType rhs) {
    return le(lhs, rhs);
}

}  // namespace ttnn
