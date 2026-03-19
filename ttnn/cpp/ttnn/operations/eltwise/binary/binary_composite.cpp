// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/binary/binary_composite.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn {

Tensor bias_gelu(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::binary::bias_gelu(
        input_tensor_a_arg,
        input_tensor_b_arg,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}

Tensor bias_gelu(
    const Tensor& input_tensor_a,
    float bias,
    const std::optional<const DataType>& /*dtype*/,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> /*post_activations*/,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> /*lhs_activations*/,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> /*rhs_activations*/,
    std::optional<bool> /*use_legacy*/,
    const std::optional<CoreRangeSet>& /*sub_core_grids*/) {
    return ttnn::gelu(
        ttnn::add(input_tensor_a, bias, std::nullopt, memory_config, optional_output_tensor),
        true,
        memory_config,
        optional_output_tensor);
}

Tensor bias_gelu(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::binary::bias_gelu(
        input_tensor_a,
        input_tensor_b,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        std::nullopt,
        sub_core_grids);
}

Tensor atan2(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, const std::optional<MemoryConfig>& memory_config) {
    return operations::binary::binary_composite_op<operations::binary::BinaryCompositeOpType::ATAN2>(
        input_tensor_a, input_tensor_b, memory_config);
}

Tensor nextafter(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, const std::optional<MemoryConfig>& memory_config) {
    return operations::binary::binary_composite_op<operations::binary::BinaryCompositeOpType::NEXTAFTER>(
        input_tensor_a, input_tensor_b, memory_config);
}

Tensor isclose(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    float rtol,
    float atol,
    const bool equal_nan,
    const std::optional<MemoryConfig>& memory_config) {
    return operations::binary::binary_composite_op_isclose<operations::binary::BinaryCompositeOpType::ISCLOSE>(
        input_tensor_a, input_tensor_b, rtol, atol, equal_nan, memory_config);
}

Tensor div_no_nan(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, const std::optional<MemoryConfig>& memory_config) {
    return operations::binary::div_like_op<operations::binary::BinaryCompositeOpType::DIV_NO_NAN>(
        input_tensor_a, input_tensor_b, memory_config);
}

Tensor div_no_nan(const Tensor& input_tensor_a, float value, const std::optional<MemoryConfig>& memory_config) {
    return operations::binary::div_like_op<operations::binary::BinaryCompositeOpType::DIV_NO_NAN>(
        input_tensor_a, value, memory_config);
}

Tensor floor_div(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, const std::optional<MemoryConfig>& memory_config) {
    return operations::binary::div_like_op<operations::binary::BinaryCompositeOpType::FLOOR_DIV>(
        input_tensor_a, input_tensor_b, memory_config);
}

Tensor floor_div(const Tensor& input_tensor_a, float value, const std::optional<MemoryConfig>& memory_config) {
    return operations::binary::div_like_op<operations::binary::BinaryCompositeOpType::FLOOR_DIV>(
        input_tensor_a, value, memory_config);
}

Tensor outer(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, const std::optional<MemoryConfig>& memory_config) {
    return operations::binary::binary_composite_op<operations::binary::BinaryCompositeOpType::OUTER>(
        input_tensor_a, input_tensor_b, memory_config);
}

Tensor polyval(
    const Tensor& input_tensor_a, const std::vector<float>& coeffs, const std::optional<MemoryConfig>& memory_config) {
    return operations::binary::binary_composite_op_polyval<operations::binary::BinaryCompositeOpType::POLYVAL>(
        input_tensor_a, coeffs, memory_config);
}

Tensor logical_left_shift(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return operations::binary::bitwise_left_shift(
        input_tensor_a_arg,
        input_tensor_b_arg,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

Tensor logical_left_shift(
    const Tensor& input_tensor,
    int32_t input_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations,
    ttsl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return operations::binary::bitwise_left_shift(
        input_tensor,
        input_b,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

}  // namespace ttnn
