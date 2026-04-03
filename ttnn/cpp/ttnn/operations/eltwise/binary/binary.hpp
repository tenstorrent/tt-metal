// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "ttnn/types.hpp"

// Macros for binary operations with identical argument signatures and implementation.
// Each macro generates function declaration(s). Use the corresponding TTNN_*_IMPL macro in .cpp.

// Tensor-Tensor binary op (calls invoke_binary_ng)
#define TTNN_BINARY_OP_TENSOR_TENSOR(NAME, OP_TYPE)                                       \
    Tensor NAME(                                                                          \
        const Tensor& lhs,                                                                \
        const Tensor& rhs,                                                                \
        const std::optional<const DataType>& output_dtype = std::nullopt,                 \
        const std::optional<MemoryConfig>& memory_config = std::nullopt,                  \
        const std::optional<Tensor>& output = std::nullopt,                               \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {}, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},  \
        const std::optional<bool>& use_legacy = std::nullopt,                             \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

// Tensor-float binary op (calls invoke_binary_ng)
#define TTNN_BINARY_OP_TENSOR_FLOAT(NAME, OP_TYPE)                                        \
    Tensor NAME(                                                                          \
        const Tensor& lhs,                                                                \
        float rhs,                                                                        \
        const std::optional<const DataType>& output_dtype = std::nullopt,                 \
        const std::optional<MemoryConfig>& memory_config = std::nullopt,                  \
        const std::optional<Tensor>& output = std::nullopt,                               \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {}, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},  \
        const std::optional<bool>& use_legacy = std::nullopt,                             \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

// Inplace binary op (Tensor-Tensor and Tensor-float, calls invoke_binary_ng with output=lhs)
#define TTNN_BINARY_OP_INPLACE(NAME, OP_TYPE)                                             \
    Tensor NAME(                                                                          \
        const Tensor& lhs,                                                                \
        const Tensor& rhs,                                                                \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {}, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},  \
        std::optional<bool> use_legacy = std::nullopt,                                    \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);                \
    Tensor NAME(                                                                          \
        const Tensor& lhs,                                                                \
        float rhs,                                                                        \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {}, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},  \
        std::optional<bool> use_legacy = std::nullopt,                                    \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

// Inplace relational binary op (Tensor-Tensor and Tensor-float, calls inplace_relational_binary)
#define TTNN_BINARY_OP_INPLACE_RELATIONAL(NAME, OP_TYPE)                                  \
    Tensor NAME(                                                                          \
        const Tensor& lhs,                                                                \
        const Tensor& rhs,                                                                \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {}, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},  \
        std::optional<bool> use_legacy = std::nullopt,                                    \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);                \
    Tensor NAME(                                                                          \
        const Tensor& lhs,                                                                \
        float rhs,                                                                        \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {}, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},  \
        std::optional<bool> use_legacy = std::nullopt,                                    \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

// Inplace binary op that calls invoke_binary_ng (logical_and_, logical_or_, logical_xor_, rsub_, bias_gelu_)
#define TTNN_BINARY_OP_INPLACE_INVOKE(NAME, OP_TYPE)                                      \
    Tensor NAME(                                                                          \
        const Tensor& lhs,                                                                \
        const Tensor& rhs,                                                                \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {}, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},  \
        std::optional<bool> use_legacy = std::nullopt,                                    \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);                \
    Tensor NAME(                                                                          \
        const Tensor& lhs,                                                                \
        float rhs,                                                                        \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {}, \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},  \
        ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},  \
        std::optional<bool> use_legacy = std::nullopt,                                    \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

namespace ttnn {
namespace operations::binary {

bool is_legacy_only(
    const Tensor& lhs,
    const auto& rhs,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations);

template <BinaryOpType binary_op_type>
Tensor where_operation_with_scalar(
    const Tensor& condition,
    const Tensor& true_false_tensor,
    unary::ScalarVariant scalar_value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace operations::binary

namespace detail {

Tensor invoke_binary_ng(
    const Tensor& lhs,
    const Tensor& rhs,
    operations::binary::BinaryOpType binary_op_type,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<bool>& fast_and_approximate_mode = false,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor invoke_binary_ng(
    const Tensor& lhs,
    float rhs,
    operations::binary::BinaryOpType binary_op_type,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<bool>& fast_and_approximate_mode = false,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor invoke_binary_ng(
    const Tensor& lhs,
    int32_t rhs,
    operations::binary::BinaryOpType binary_op_type,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy,
    const std::optional<bool>& fast_and_approximate_mode = false,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace detail

TTNN_BINARY_OP_TENSOR_TENSOR(add, ADD)
TTNN_BINARY_OP_TENSOR_FLOAT(add, ADD)
TTNN_BINARY_OP_INPLACE(add_, ADD)
TTNN_BINARY_OP_TENSOR_TENSOR(subtract, SUB)
TTNN_BINARY_OP_TENSOR_FLOAT(subtract, SUB)
TTNN_BINARY_OP_INPLACE(subtract_, SUB)
TTNN_BINARY_OP_TENSOR_TENSOR(eq, EQ)
Tensor eq(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor eq(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);
TTNN_BINARY_OP_TENSOR_TENSOR(ne, NE)
Tensor ne(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor ne(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);
TTNN_BINARY_OP_TENSOR_TENSOR(ge, GE)
Tensor ge(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor ge(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);
TTNN_BINARY_OP_TENSOR_TENSOR(gt, GT)
Tensor gt(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor gt(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);
TTNN_BINARY_OP_TENSOR_TENSOR(le, LE)
Tensor le(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor le(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);
TTNN_BINARY_OP_TENSOR_TENSOR(lt, LT)
Tensor lt(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor lt(
    float lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt);
TTNN_BINARY_OP_TENSOR_TENSOR(logical_and, LOGICAL_AND)
TTNN_BINARY_OP_TENSOR_FLOAT(logical_and, LOGICAL_AND)
TTNN_BINARY_OP_TENSOR_TENSOR(logical_or, LOGICAL_OR)
TTNN_BINARY_OP_TENSOR_FLOAT(logical_or, LOGICAL_OR)
TTNN_BINARY_OP_TENSOR_TENSOR(logical_xor, LOGICAL_XOR)
TTNN_BINARY_OP_TENSOR_FLOAT(logical_xor, LOGICAL_XOR)
TTNN_BINARY_OP_TENSOR_TENSOR(ldexp, LDEXP)
TTNN_BINARY_OP_TENSOR_FLOAT(ldexp, LDEXP)
TTNN_BINARY_OP_INPLACE(ldexp_, LDEXP)
TTNN_BINARY_OP_TENSOR_TENSOR(logaddexp, LOGADDEXP)
TTNN_BINARY_OP_TENSOR_FLOAT(logaddexp, LOGADDEXP)
TTNN_BINARY_OP_INPLACE(logaddexp_, LOGADDEXP)
TTNN_BINARY_OP_TENSOR_TENSOR(logaddexp2, LOGADDEXP2)
TTNN_BINARY_OP_TENSOR_FLOAT(logaddexp2, LOGADDEXP2)
TTNN_BINARY_OP_INPLACE(logaddexp2_, LOGADDEXP2)
TTNN_BINARY_OP_TENSOR_TENSOR(squared_difference, SQUARED_DIFFERENCE)
TTNN_BINARY_OP_TENSOR_FLOAT(squared_difference, SQUARED_DIFFERENCE)
TTNN_BINARY_OP_INPLACE(squared_difference_, SQUARED_DIFFERENCE)
Tensor divide(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor divide(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor divide_(
    const Tensor& lhs,
    const Tensor& rhs,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    std::optional<bool> fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor divide_(
    const Tensor& lhs,
    float rhs,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    std::optional<bool> fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor multiply(
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor multiply(
    const Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<bool>& fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor multiply(const Tensor& lhs, const Tensor& rhs, bool fast_and_approximate_mode);
Tensor multiply(const Tensor& lhs, float rhs, bool fast_and_approximate_mode);
Tensor multiply_(
    const Tensor& lhs,
    const Tensor& rhs,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    std::optional<bool> fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
Tensor multiply_(
    const Tensor& lhs,
    float rhs,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    std::optional<bool> fast_and_approximate_mode = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
TTNN_BINARY_OP_INPLACE_RELATIONAL(gt_, GT)
TTNN_BINARY_OP_INPLACE_RELATIONAL(ge_, GE)
TTNN_BINARY_OP_INPLACE_RELATIONAL(le_, LE)
TTNN_BINARY_OP_INPLACE_RELATIONAL(lt_, LT)
TTNN_BINARY_OP_INPLACE_INVOKE(logical_and_, LOGICAL_AND)
TTNN_BINARY_OP_INPLACE_INVOKE(logical_or_, LOGICAL_OR)
TTNN_BINARY_OP_INPLACE_INVOKE(logical_xor_, LOGICAL_XOR)
TTNN_BINARY_OP_INPLACE_RELATIONAL(eq_, EQ)
TTNN_BINARY_OP_INPLACE_RELATIONAL(ne_, NE)
TTNN_BINARY_OP_INPLACE_INVOKE(rsub_, RSUB)
TTNN_BINARY_OP_INPLACE_INVOKE(bias_gelu_, BIAS_GELU)
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
TTNN_BINARY_OP_TENSOR_TENSOR(logical_right_shift, LOGICAL_RIGHT_SHIFT)
TTNN_BINARY_OP_TENSOR_FLOAT(logical_right_shift, LOGICAL_RIGHT_SHIFT)
TTNN_BINARY_OP_TENSOR_TENSOR(xlogy, XLOGY)
TTNN_BINARY_OP_TENSOR_FLOAT(xlogy, XLOGY)
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

#undef TTNN_BINARY_OP_TENSOR_TENSOR
#undef TTNN_BINARY_OP_TENSOR_FLOAT
#undef TTNN_BINARY_OP_INPLACE
#undef TTNN_BINARY_OP_INPLACE_RELATIONAL
#undef TTNN_BINARY_OP_INPLACE_INVOKE
