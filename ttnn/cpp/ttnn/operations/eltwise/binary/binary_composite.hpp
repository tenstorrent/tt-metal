// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <vector>

#include <tt_stl/span.hpp>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/binary/device/binary_composite_op.hpp"
#include "ttnn/operations/eltwise/binary/device/binary_device_operation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn {

namespace operations::binary {

/**
 * @brief Performs element-wise power operation on the input with the exponent.
 * When exponent is Tensor, the supported dtypes are float32 and bfloat16.
 * The tested range for the input is (-30,30) and for the exponent is (-20, 20).
 *
 * @param input The input tensor, i.e the base.
 * @param exponent The exponent
 * @return The result tensor
 */
Tensor pow(
    const Tensor& input_tensor,
    int32_t exponent,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor pow(
    const Tensor& input_a,
    float exponent,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor pow(
    float input_a,
    const Tensor& exponent,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor pow(
    const Tensor& input_tensor,
    const Tensor& exponent,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

template <BinaryCompositeOpType binary_comp_op_type>
Tensor binary_composite_op(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return OpHandler<binary_comp_op_type>::handle(input_tensor_a, input_tensor_b, memory_config);
}

template <BinaryCompositeOpType binary_comp_op_type>
Tensor binary_composite_op_isclose(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    float rtol,
    float atol,
    const bool equal_nan,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return OpHandler<binary_comp_op_type>::handle(input_tensor_a, input_tensor_b, rtol, atol, equal_nan, memory_config);
}

template <BinaryCompositeOpType binary_comp_op_type>
Tensor div_like_op(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return OpHandler<binary_comp_op_type>::handle(input_tensor_a, input_tensor_b, memory_config);
}

template <BinaryCompositeOpType binary_comp_op_type>
Tensor div_like_op(
    const Tensor& input_tensor_a, float value, const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return OpHandler<binary_comp_op_type>::handle(input_tensor_a, value, memory_config);
}

Tensor div(
    const Tensor& input_a,
    const Tensor& input_b,
    bool fast_and_approximate_mode = false,
    const std::optional<std::string>& rounding_mode = std::nullopt,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor div(
    const Tensor& input,
    float value,
    bool fast_and_approximate_mode = false,
    const std::optional<std::string>& rounding_mode = std::nullopt,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    std::optional<Tensor> output = std::nullopt,
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor bias_gelu(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor bias_gelu(
    const ttnn::Tensor& input_tensor_a,
    float bias,
    const std::optional<const DataType>& /*dtype*/ = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> /*post_activations*/ = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> /*lhs_activations*/ = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> /*rhs_activations*/ = {},
    std::optional<bool> /*use_legacy*/ = std::nullopt,
    const std::optional<CoreRangeSet>& /*sub_core_grids*/ = std::nullopt);

Tensor bias_gelu(
    const Tensor& input_tensor_a,
    float scalar,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {});

template <BinaryCompositeOpType binary_comp_op_type>
Tensor binary_composite_op_polyval(
    const Tensor& input_tensor_a,
    const std::vector<float>& coeffs,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return OpHandler<binary_comp_op_type>::handle(input_tensor_a, coeffs, memory_config);
}

Tensor fmod(
    const Tensor& input_a,
    const Tensor& input_b,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor fmod(
    const Tensor& input_tensor,
    float scalar,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor remainder(
    const Tensor& input_a,
    const Tensor& input_b,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor remainder(
    const Tensor& input_tensor,
    float scalar,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor lcm(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor gcd(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor maximum(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor maximum(
    const Tensor& input_a,
    unary::ScalarVariant value,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor minimum(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor minimum(
    const Tensor& input_a,
    unary::ScalarVariant value,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor prelu(
    const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

Tensor prelu(
    const Tensor& input_tensor,
    const std::array<float, 1>& weight,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

Tensor prelu(
    const Tensor& input_tensor, float weight, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

Tensor rsub(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor rsub(
    const Tensor& input_tensor,
    float input_b,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor bitwise_and(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor bitwise_and(
    const Tensor& input_tensor,
    int32_t input_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor bitwise_or(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor bitwise_or(
    const Tensor& input_tensor,
    int32_t input_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor bitwise_xor(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor bitwise_xor(
    const Tensor& input_tensor,
    int32_t input_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor bitwise_left_shift(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor bitwise_left_shift(
    const Tensor& input_tensor,
    int32_t input_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor bitwise_right_shift(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor bitwise_right_shift(
    const Tensor& input_tensor,
    int32_t input_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

}  // namespace operations::binary

Tensor bias_gelu(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor bias_gelu(
    const Tensor& input_tensor_a,
    float bias,
    const std::optional<const DataType>& /*dtype*/ = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> /*post_activations*/ = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> /*lhs_activations*/ = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> /*rhs_activations*/ = {},
    std::optional<bool> /*use_legacy*/ = std::nullopt,
    const std::optional<CoreRangeSet>& /*sub_core_grids*/ = std::nullopt);

Tensor bias_gelu(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

using operations::binary::bitwise_and;
using operations::binary::bitwise_left_shift;
using operations::binary::bitwise_or;
using operations::binary::bitwise_right_shift;
using operations::binary::bitwise_xor;
using operations::binary::div;
using operations::binary::fmod;
using operations::binary::gcd;
using operations::binary::lcm;
using operations::binary::maximum;
using operations::binary::minimum;
using operations::binary::pow;
using operations::binary::prelu;
using operations::binary::remainder;
using operations::binary::rsub;

Tensor atan2(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

Tensor nextafter(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

Tensor isclose(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    float rtol,
    float atol,
    bool equal_nan,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

Tensor div_no_nan(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

Tensor div_no_nan(
    const Tensor& input_tensor_a, float value, const std::optional<MemoryConfig>& memory_config = std::nullopt);

Tensor floor_div(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

Tensor floor_div(
    const Tensor& input_tensor_a, float value, const std::optional<MemoryConfig>& memory_config = std::nullopt);

Tensor outer(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

Tensor polyval(
    const Tensor& input_tensor_a,
    const std::vector<float>& coeffs,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

Tensor logical_left_shift(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

Tensor logical_left_shift(
    const Tensor& input_tensor,
    int32_t input_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt);

}  // namespace ttnn
