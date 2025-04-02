// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/binary_ng/types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

ttnn::Tensor typecast_to(ttnn::DataType dtype, const ttnn::Tensor& input);
bool needs_typecast_to_bfloat16(const ttnn::DataType input);

namespace ttnn::operations::binary_ng {

template <BinaryOpType binary_op_type>
struct BinaryNg {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::UnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::UnaryWithParam> rhs_activations = {},
        tt::stl::Span<const unary::UnaryWithParam> post_activations = {});

    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor_a,
        float scalar,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::UnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::UnaryWithParam> rhs_activations = {},
        tt::stl::Span<const unary::UnaryWithParam> post_activations = {});
};

template <BinaryOpType binary_op_type>
struct BinaryNgBitwise {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);

    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor_a,
        float scalar,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct InplaceBinaryNg {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        tt::stl::Span<const unary::UnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::UnaryWithParam> rhs_activations = {},
        tt::stl::Span<const unary::UnaryWithParam> post_activations = {});

    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        float scalar,
        tt::stl::Span<const unary::UnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::UnaryWithParam> rhs_activations = {},
        tt::stl::Span<const unary::UnaryWithParam> post_activations = {});
};

}  // namespace ttnn::operations::binary_ng

namespace ttnn::experimental {
constexpr auto add = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::add",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::ADD>>();

constexpr auto sub = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::sub",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::SUB>>();

constexpr auto rsub = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::rsub",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::RSUB>>();

constexpr auto mul = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::mul",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::MUL>>();

constexpr auto div = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::div",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::DIV>>();

constexpr auto eq = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::eq",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::EQ>>();

constexpr auto ne = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::ne",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::NE>>();

constexpr auto gt = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::gt",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::GT>>();

constexpr auto gte = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::gte",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::GTE>>();

constexpr auto lt = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::lt",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::LT>>();

constexpr auto lte = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::lte",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::LTE>>();

constexpr auto pow = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::pow",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::POWER>>();

constexpr auto squared_difference = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::squared_difference",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::SQUARED_DIFFERENCE>>();

constexpr auto bias_gelu = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::bias_gelu",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::BIAS_GELU>>();

constexpr auto logical_and = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::logical_and",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::LOGICAL_AND>>();

constexpr auto logical_or = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::logical_or",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::LOGICAL_OR>>();

constexpr auto logical_xor = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::logical_xor",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::LOGICAL_XOR>>();

constexpr auto ldexp = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::ldexp",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::LDEXP>>();

constexpr auto logaddexp = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::logaddexp",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::LOGADDEXP>>();

constexpr auto logaddexp2 = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::logaddexp2",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::LOGADDEXP2>>();

constexpr auto bitwise_and = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::bitwise_and",
    ttnn::operations::binary_ng::BinaryNgBitwise<operations::binary_ng::BinaryOpType::BITWISE_AND>>();

constexpr auto bitwise_xor = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::bitwise_xor",
    ttnn::operations::binary_ng::BinaryNgBitwise<operations::binary_ng::BinaryOpType::BITWISE_XOR>>();

constexpr auto bitwise_or = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::bitwise_or",
    ttnn::operations::binary_ng::BinaryNgBitwise<operations::binary_ng::BinaryOpType::BITWISE_OR>>();

constexpr auto bitwise_left_shift = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::bitwise_left_shift",
    ttnn::operations::binary_ng::BinaryNgBitwise<operations::binary_ng::BinaryOpType::LEFT_SHIFT>>();

constexpr auto bitwise_right_shift = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::bitwise_right_shift",
    ttnn::operations::binary_ng::BinaryNgBitwise<operations::binary_ng::BinaryOpType::RIGHT_SHIFT>>();

constexpr auto add_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::add_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::ADD>>();

constexpr auto sub_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::sub_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::SUB>>();

constexpr auto mul_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::mul_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::MUL>>();

constexpr auto div_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::div_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::DIV>>();

constexpr auto rsub_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::rsub_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::RSUB>>();

constexpr auto pow_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::pow_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::POWER>>();

constexpr auto gt_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::gt_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::GT>>();

constexpr auto lt_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::lt_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::LT>>();

constexpr auto lte_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::lte_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::LTE>>();

constexpr auto gte_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::gte_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::GTE>>();

constexpr auto eq_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::eq_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::EQ>>();

constexpr auto ne_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::ne_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::NE>>();

constexpr auto squared_difference_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::squared_difference_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::SQUARED_DIFFERENCE>>();

constexpr auto bias_gelu_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::bias_gelu_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::BIAS_GELU>>();

constexpr auto logical_and_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::logical_and_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::LOGICAL_AND>>();

constexpr auto logical_or_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::logical_or_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::LOGICAL_OR>>();

constexpr auto logical_xor_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::logical_xor_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::LOGICAL_XOR>>();

constexpr auto ldexp_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::ldexp_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::LDEXP>>();

constexpr auto logaddexp_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::logaddexp_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::LOGADDEXP>>();

constexpr auto logaddexp2_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::logaddexp2_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::LOGADDEXP2>>();

}  // namespace ttnn::experimental
