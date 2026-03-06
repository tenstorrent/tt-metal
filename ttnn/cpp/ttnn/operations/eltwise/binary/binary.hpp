
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
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::BinaryOperation<Op>::invoke(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}

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
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::BinaryOperation<Op>::invoke(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}

inline Tensor add(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::ADD>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor add(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::ADD>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor add_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::ADD>::invoke(
        lhs, rhs, post_activations, lhs_activations, rhs_activations, use_legacy, sub_core_grids);
}
inline Tensor add_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::ADD>::invoke(
        lhs, rhs, post_activations, lhs_activations, rhs_activations, use_legacy, sub_core_grids);
}
inline Tensor subtract(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::SUB>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor subtract(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::SUB>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor subtract_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::SUB>::invoke(
        lhs, rhs, post_activations, lhs_activations, rhs_activations, use_legacy, sub_core_grids);
}
inline Tensor subtract_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::SUB>::invoke(
        lhs, rhs, post_activations, lhs_activations, rhs_activations, use_legacy, sub_core_grids);
}
inline Tensor eq(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::EQ>::invoke(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor eq(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::EQ>::invoke(
        lhs,
        rhs,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor eq(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::EQ>::invoke(
        lhs, rhs, dtype, memory_config, output);
}
inline Tensor ne(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::NE>::invoke(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor ne(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::NE>::invoke(
        lhs,
        rhs,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor ne(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::NE>::invoke(
        lhs, rhs, dtype, memory_config, output);
}
inline Tensor ge(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::GE>::invoke(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor ge(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::GE>::invoke(
        lhs,
        rhs,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor ge(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::GE>::invoke(
        lhs, rhs, dtype, memory_config, output);
}
inline Tensor gt(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::GT>::invoke(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor gt(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::GT>::invoke(
        lhs,
        rhs,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor gt(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::GT>::invoke(
        lhs, rhs, dtype, memory_config, output);
}
inline Tensor le(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::LE>::invoke(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor le(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::LE>::invoke(
        lhs,
        rhs,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor le(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::LE>::invoke(
        lhs, rhs, dtype, memory_config, output);
}
inline Tensor lt(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::LT>::invoke(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor lt(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::LT>::invoke(
        lhs,
        rhs,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor lt(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt) {
    return operations::binary::RelationalBinary<operations::binary::BinaryOpType::LT>::invoke(
        lhs, rhs, dtype, memory_config, output);
}
inline Tensor logical_and(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::LOGICAL_AND>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor logical_and(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::LOGICAL_AND>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor logical_or(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::LOGICAL_OR>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor logical_or(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::LOGICAL_OR>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor logical_xor(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::LOGICAL_XOR>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor logical_xor(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::LOGICAL_XOR>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor ldexp(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::LDEXP>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor ldexp(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::LDEXP>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor ldexp_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::LDEXP>::invoke(
        lhs, rhs, post_activations, lhs_activations, rhs_activations, use_legacy, sub_core_grids);
}
inline Tensor ldexp_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::LDEXP>::invoke(
        lhs, rhs, post_activations, lhs_activations, rhs_activations, use_legacy, sub_core_grids);
}
inline Tensor logaddexp(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::LOGADDEXP>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor logaddexp(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::LOGADDEXP>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor logaddexp_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::LOGADDEXP>::invoke(
        lhs, rhs, post_activations, lhs_activations, rhs_activations, use_legacy, sub_core_grids);
}
inline Tensor logaddexp_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::LOGADDEXP>::invoke(
        lhs, rhs, post_activations, lhs_activations, rhs_activations, use_legacy, sub_core_grids);
}
inline Tensor logaddexp2(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::LOGADDEXP2>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor logaddexp2(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::LOGADDEXP2>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor logaddexp2_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::LOGADDEXP2>::invoke(
        lhs, rhs, post_activations, lhs_activations, rhs_activations, use_legacy, sub_core_grids);
}
inline Tensor logaddexp2_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::LOGADDEXP2>::invoke(
        lhs, rhs, post_activations, lhs_activations, rhs_activations, use_legacy, sub_core_grids);
}
inline Tensor squared_difference(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::SQUARED_DIFFERENCE>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor squared_difference(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::SQUARED_DIFFERENCE>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor squared_difference_(
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::SQUARED_DIFFERENCE>::invoke(
        lhs, rhs, post_activations, lhs_activations, rhs_activations, use_legacy, sub_core_grids);
}
inline Tensor squared_difference_(
    const Tensor& lhs,
    float rhs,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::SQUARED_DIFFERENCE>::invoke(
        lhs, rhs, post_activations, lhs_activations, rhs_activations, use_legacy, sub_core_grids);
}
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
inline Tensor logical_right_shift(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::LOGICAL_RIGHT_SHIFT>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor logical_right_shift(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::LOGICAL_RIGHT_SHIFT>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor xlogy(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::XLOGY>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor xlogy(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return binary_op<operations::binary::BinaryOpType::XLOGY>(
        lhs,
        rhs,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
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
