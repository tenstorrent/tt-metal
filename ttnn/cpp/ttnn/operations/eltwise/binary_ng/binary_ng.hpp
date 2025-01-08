
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/binary_ng/types.hpp"
#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::binary_ng {

template <BinaryOpType binary_op_type>
struct BinaryNg {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::UnaryOpType> lhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> rhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> post_activations = {});

    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::UnaryOpType> lhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> rhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> post_activations = {});

    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor_a,
        float scalar,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::UnaryOpType> lhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> rhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> post_activations = {});

    static Tensor invoke(
        const Tensor& input_tensor_a,
        float scalar,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::UnaryOpType> lhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> rhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> post_activations = {});
};

template <BinaryOpType binary_op_type>
struct InplaceBinaryNg {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        tt::stl::Span<const unary::UnaryOpType> lhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> rhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> post_activations = {});

    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const float scalar,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        tt::stl::Span<const unary::UnaryOpType> lhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> rhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> post_activations = {});

    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        tt::stl::Span<const unary::UnaryOpType> lhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> rhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> post_activations = {});

    static Tensor invoke(
        const Tensor& input_tensor,
        const float scalar,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        tt::stl::Span<const unary::UnaryOpType> lhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> rhs_activations = {},
        tt::stl::Span<const unary::UnaryOpType> post_activations = {});
};

}  // namespace ttnn::operations::binary_ng

inline Tensor typecast_to(DataType dtype, const Tensor& input) {
    return input.get_dtype() == dtype ? input : ttnn::typecast(input, dtype);
}

namespace ttnn::experimental {
constexpr auto add = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::add",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::ADD>>();

constexpr auto sub = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::sub",
    ttnn::operations::binary_ng::BinaryNg<operations::binary_ng::BinaryOpType::SUB>>();

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

constexpr auto add_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::add_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::ADD>>();

constexpr auto sub_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::sub_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::SUB>>();

constexpr auto mul_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::mul_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::MUL>>();

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

constexpr auto logical_and_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::logical_and_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::LOGICAL_AND>>();

constexpr auto logical_or_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::logical_or_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::LOGICAL_OR>>();

constexpr auto logical_xor_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::logical_xor_",
    ttnn::operations::binary_ng::InplaceBinaryNg<operations::binary_ng::BinaryOpType::LOGICAL_XOR>>();

}  // namespace ttnn::experimental
