
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations::binary {

bool is_legacy_only(
    const Tensor& lhs,
    const auto& rhs,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations);

template <BinaryOpType binary_op_type>
struct BinaryOperation {
    static Tensor invoke(
        const Tensor& lhs,
        const Tensor& rhs,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        const std::optional<bool>& use_legacy = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static Tensor invoke(
        const ttnn::Tensor& lhs,
        float rhs,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        const std::optional<bool>& use_legacy = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct BinaryOperationWithFastApprox {
    static Tensor invoke(
        const Tensor& lhs,
        const Tensor& rhs,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        const std::optional<bool>& use_legacy = std::nullopt,
        const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static Tensor invoke(
        const ttnn::Tensor& lhs,
        float rhs,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        const std::optional<bool>& use_legacy = std::nullopt,
        const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct MulOperationWithFastApprox {
    static Tensor invoke(
        const Tensor& lhs,
        const Tensor& rhs,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        const std::optional<bool>& use_legacy = std::nullopt,
        const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static Tensor invoke(
        const ttnn::Tensor& lhs,
        float rhs,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        const std::optional<bool>& use_legacy = std::nullopt,
        const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    // Simplified overloads for fast_and_approximate_mode
    static Tensor invoke(const Tensor& lhs, const Tensor& rhs, bool fast_and_approximate_mode);

    static Tensor invoke(const Tensor& lhs, float rhs, bool fast_and_approximate_mode);
};

template <BinaryOpType binary_op_type>
struct RelationalBinary {
    static Tensor invoke(
        const Tensor& lhs,
        const Tensor& rhs,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        const std::optional<bool>& use_legacy = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static Tensor invoke(
        const ttnn::Tensor& lhs,
        float rhs,
        const std::optional<const DataType>& dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        const std::optional<bool>& use_legacy = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    // rhs - tensor combination not available on Pytorch for this op
    static Tensor invoke(
        float lhs,
        const ttnn::Tensor& rhs,
        const std::optional<const DataType>& dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct InplaceRelationalBinary {
    static Tensor invoke(
        const Tensor& lhs,
        const Tensor& rhs,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static Tensor invoke(
        const Tensor& lhs,
        float rhs,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct InplaceLogicalBinary {
    static Tensor invoke(
        const Tensor& lhs,
        const Tensor& rhs,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static Tensor invoke(
        const Tensor& lhs,
        float rhs,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct InplaceBinaryOperation {
    static Tensor invoke(
        const Tensor& lhs,
        const Tensor& rhs,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static Tensor invoke(
        const Tensor& lhs,
        float rhs,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct InplaceBinaryOperationWithFastApprox {
    static Tensor invoke(
        const Tensor& lhs,
        const Tensor& rhs,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt,
        std::optional<bool> fast_and_approximate_mode = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static Tensor invoke(
        const Tensor& lhs,
        float rhs,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt,
        std::optional<bool> fast_and_approximate_mode = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct InplaceMulOperationWithFastApprox {
    static Tensor invoke(
        const Tensor& lhs,
        const Tensor& rhs,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt,
        std::optional<bool> fast_and_approximate_mode = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static Tensor invoke(
        const Tensor& lhs,
        float rhs,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt,
        std::optional<bool> fast_and_approximate_mode = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct BinaryOperationSfpu {
    static Tensor invoke(
        const Tensor& lhs,
        const Tensor& rhs,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct BinaryOperationAddalpha {
    static Tensor invoke(
        const Tensor& lhs,
        const Tensor& rhs,
        float alpha,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct BinaryOperationSubalpha {
    static Tensor invoke(
        const Tensor& lhs,
        const Tensor& rhs,
        float alpha,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct BinaryOperationHypot {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct WhereOperationWithScalar {
    // For TTS / TST variant
    static Tensor invoke(
        const Tensor& condition,
        const Tensor& true_false_tensor,  // For TTS variant, true_tensor; For TST variant, false_tensor
        unary::ScalarVariant scalar_value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

}  // namespace operations::binary

// Binary public API uses the same two-layer pattern as ternary: free function -> implementation
// (detail::invoke_binary_ng). BinaryOperation<Op> and related structs remain for internal
// call-sites (composite ops, inplace, nanobind, etc.) and may be migrated away in a follow-up.

template <operations::binary::BinaryOpType Op>
Tensor binary_op(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

template <operations::binary::BinaryOpType Op>
Tensor binary_op(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor add(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor add(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor add_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor add_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor subtract(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor subtract(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor subtract_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor subtract_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor eq(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor eq(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor eq(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);
Tensor ne(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor ne(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor ne(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);
Tensor ge(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor ge(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor ge(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);
Tensor gt(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor gt(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor gt(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);
Tensor le(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor le(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor le(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);
Tensor lt(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor lt(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor lt(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);
Tensor logical_and(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logical_and(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logical_or(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logical_or(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logical_xor(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logical_xor(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor ldexp(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor ldexp(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor ldexp_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor ldexp_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logaddexp(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logaddexp(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logaddexp_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logaddexp_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logaddexp2(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logaddexp2(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logaddexp2_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logaddexp2_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor squared_difference(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor squared_difference(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor squared_difference_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor squared_difference_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor divide(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor divide(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor divide_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    std::optional<bool> fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor divide_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    std::optional<bool> fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor multiply(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor multiply(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor multiply(const Tensor& lhs, const Tensor& rhs, bool fast_and_approximate_mode);
Tensor multiply(const Tensor& lhs, float rhs, bool fast_and_approximate_mode);
Tensor multiply_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    std::optional<bool> fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor multiply_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    std::optional<bool> fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor gt_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor gt_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor ge_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor ge_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor le_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor le_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor lt_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor lt_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logical_and_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logical_and_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logical_or_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logical_or_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logical_xor_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logical_xor_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor eq_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor eq_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor ne_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor ne_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor rsub_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor rsub_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor bias_gelu_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor bias_gelu_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor addalpha(
    const Tensor& lhs,
    const Tensor& rhs,
    float alpha,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);
Tensor subalpha(
    const Tensor& lhs,
    const Tensor& rhs,
    float alpha,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);
Tensor logical_right_shift(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor logical_right_shift(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor xlogy(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor xlogy(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor hypot(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

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
